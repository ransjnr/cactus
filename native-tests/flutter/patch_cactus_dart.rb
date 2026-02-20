src_path = ARGV[0]
dst_path = ARGV[1]

src = File.read(src_path)
src.gsub!('final malloc = _MallocAllocator();', 'const malloc = _MallocAllocator();')
src.gsub!('class _MallocAllocator implements Allocator {',
          "class _MallocAllocator implements Allocator {\n  const _MallocAllocator();")
src += "\nfinal class Utf8 extends Opaque {}\n"
File.write(dst_path, src)
