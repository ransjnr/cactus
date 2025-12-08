#import "AppDelegate.h"
#import <sys/xattr.h>
#import <sys/resource.h>

extern int test_kernel_main();
extern int test_graph_main();
extern int test_kv_cache_main();
extern int test_engine_main();
extern int test_performance_main();

@implementation AppDelegate

- (void)removeExtendedAttributes:(NSString *)path {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    BOOL isDirectory;

    if ([fileManager fileExistsAtPath:path isDirectory:&isDirectory]) {
        // Remove ALL extended attributes from this file/directory
        const char *cPath = [path UTF8String];
        ssize_t size = listxattr(cPath, NULL, 0, XATTR_NOFOLLOW);
        if (size > 0) {
            char *attrList = (char *)malloc(size);
            if (attrList) {
                listxattr(cPath, attrList, size, XATTR_NOFOLLOW);
                char *attr = attrList;
                while (attr < attrList + size) {
                    removexattr(cPath, attr, XATTR_NOFOLLOW);
                    attr += strlen(attr) + 1;
                }
                free(attrList);
            }
        }

        // If it's a directory, recurse into contents
        if (isDirectory) {
            NSArray *contents = [fileManager contentsOfDirectoryAtPath:path error:nil];
            for (NSString *item in contents) {
                NSString *itemPath = [path stringByAppendingPathComponent:item];
                [self removeExtendedAttributes:itemPath];
            }
        }
    }
}

- (NSString *)copyModelToDocuments:(NSString *)modelName fromBundle:(NSString *)bundlePath {
    NSString *sourceModelPath = [NSString stringWithFormat:@"%@/%@", bundlePath, modelName];
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = paths[0];
    NSString *destModelPath = [NSString stringWithFormat:@"%@/%@", documentsDirectory, modelName];

    NSFileManager *fileManager = [NSFileManager defaultManager];

    // Remove existing if present
    if ([fileManager fileExistsAtPath:destModelPath]) {
        [fileManager removeItemAtPath:destModelPath error:nil];
    }

    // Copy model directory to Documents
    NSError *error = nil;
    if ([fileManager copyItemAtPath:sourceModelPath toPath:destModelPath error:&error]) {
        printf("Copied %s to Documents directory\n", [modelName UTF8String]);

        // Remove extended attributes that might interfere with mmap
        [self removeExtendedAttributes:destModelPath];
        [self fixFilePermissions:destModelPath];
        printf("Fixed permissions and attributes for %s\n", [modelName UTF8String]);

        return destModelPath;
    } else {
        printf("Failed to copy %s: %s\n", [modelName UTF8String], [[error description] UTF8String]);
        return sourceModelPath; // Fallback to bundle path
    }
}

- (void)fixFilePermissions:(NSString *)path {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    BOOL isDirectory;

    if ([fileManager fileExistsAtPath:path isDirectory:&isDirectory]) {
        // Set permissions: 0755 for directories (rwxr-xr-x), 0644 for files (rw-r--r--)
        NSNumber *permissions = isDirectory ? @(0755) : @(0644);
        NSDictionary *attrs = @{NSFilePosixPermissions: permissions};
        NSError *error = nil;
        if (![fileManager setAttributes:attrs ofItemAtPath:path error:&error]) {
            printf("Warning: Failed to set permissions for %s: %s\n",
                   [path UTF8String],
                   [[error description] UTF8String]);
        }

        // If it's a directory, recurse into contents
        if (isDirectory) {
            NSArray *contents = [fileManager contentsOfDirectoryAtPath:path error:nil];
            for (NSString *item in contents) {
                NSString *itemPath = [path stringByAppendingPathComponent:item];
                [self fixFilePermissions:itemPath];
            }
        }
    }
}

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
#if !TARGET_OS_SIMULATOR
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = paths[0];
    NSString *logPath = [documentsDirectory stringByAppendingPathComponent:@"cactus_test.log"];

    freopen([logPath UTF8String], "w", stdout);
    freopen([logPath UTF8String], "a", stderr);

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
#endif

    // Increase file descriptor limit
    struct rlimit limit;
    limit.rlim_cur = 4096;
    limit.rlim_max = 4096;
    if (setrlimit(RLIMIT_NOFILE, &limit) != 0) {
        printf("Warning: Failed to increase file descriptor limit\n");
    } else {
        printf("File descriptor limit increased to %llu\n", limit.rlim_cur);
    }

    // Get absolute path to app bundle
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];

    // Update environment variables to use paths in Documents directory (writable, mmap-able)
    const char *modelDir = getenv("CACTUS_TEST_MODEL");
    const char *transcribeModelDir = getenv("CACTUS_TEST_TRANSCRIBE_MODEL");

    if (modelDir) {
        NSString *modelPath = [self copyModelToDocuments:[NSString stringWithUTF8String:modelDir] fromBundle:bundlePath];
        setenv("CACTUS_TEST_MODEL", [modelPath UTF8String], 1);
    }

    if (transcribeModelDir) {
        NSString *transcribePath = [self copyModelToDocuments:[NSString stringWithUTF8String:transcribeModelDir] fromBundle:bundlePath];
        setenv("CACTUS_TEST_TRANSCRIBE_MODEL", [transcribePath UTF8String], 1);
    }

    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        test_kernel_main();
        test_graph_main();
        test_kv_cache_main();
        test_engine_main();
        test_performance_main();
        exit(0);
    });

    return YES;
}

- (UISceneConfiguration *)application:(UIApplication *)application configurationForConnectingSceneSession:(UISceneSession *)connectingSceneSession options:(UISceneConnectionOptions *)options {
    return [[UISceneConfiguration alloc] initWithName:@"Default Configuration" sessionRole:connectingSceneSession.role];
}

- (void)application:(UIApplication *)application didDiscardSceneSessions:(NSSet<UISceneSession *> *)sceneSessions {}

@end
