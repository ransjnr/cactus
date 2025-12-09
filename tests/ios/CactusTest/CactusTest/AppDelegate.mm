#import "AppDelegate.h"
#import <sys/resource.h>
#import <unistd.h>

extern int test_kernel_main();
extern int test_graph_main();
extern int test_kv_cache_main();
extern int test_engine_main();
extern int test_performance_main();

@implementation AppDelegate

- (void)copyModelFromBundle:(NSString *)bundlePath toDocuments:(const char *)modelDir {
    if (!modelDir) return;

    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *modelName = [NSString stringWithUTF8String:modelDir];
    NSString *sourceModelPath = [NSString stringWithFormat:@"%@/%@", bundlePath, modelName];

    if ([fileManager fileExistsAtPath:modelName]) {
        [fileManager removeItemAtPath:modelName error:nil];
    }

    [fileManager copyItemAtPath:sourceModelPath toPath:modelName error:nil];
}

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = paths[0];
    chdir([documentsDirectory UTF8String]);

#if !TARGET_OS_SIMULATOR
    freopen("cactus_test.log", "w", stdout);
    freopen("cactus_test.log", "a", stderr);
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
#endif

    struct rlimit limit = {4096, 4096};
    setrlimit(RLIMIT_NOFILE, &limit);

    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    [self copyModelFromBundle:bundlePath toDocuments:getenv("CACTUS_TEST_MODEL")];
    [self copyModelFromBundle:bundlePath toDocuments:getenv("CACTUS_TEST_TRANSCRIBE_MODEL")];

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
