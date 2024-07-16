---
title: hotspot 内存布局和源码分析
date: 2024-01-19 08:50:03
tags:
---
## 阅读jdk17中HotSpot源码
* 内存类型
HotSpot将内部的对象分为StackObj和CHeapObj分割了C的堆区和栈区，对于Java用户使用的堆HotSpot通过mmap的方式mmap出一块具体的内存负责堆区的内存分配，这需要JVM初始启动时指定参数，配合GC参数在启动阶段分配内存。在HotSpot中抽象了CollectedHeap代表Java用户的堆区，在开启NMT的情况下为了方便分析将内存指定了对应标签方便用户分析。一下是NMT的标签这部分是为了管理JVM内存皴法在CHeapObj内部。
```c++
enum MemoryType {
  // Memory type by sub systems. It occupies lower byte.
  mtJavaHeap,          // Java heap
  mtClass,             // memory class for Java classes
  mtThread,            // memory for thread objects
  mtThreadStack,
  mtCode,              // memory for generated code
  mtGC,                // memory for GC
  mtCompiler,          // memory for compiler
  mtInternal,          // memory used by VM, but does not belong to
                       // any of above categories, and not used for
                       // native memory tracking
  mtOther,             // memory not used by VM
  mtSymbol,            // symbol
  mtNMT,               // memory used by native memory tracking
  mtClassShared,       // class data sharing
  mtChunk,             // chunk that holds content of arenas
  mtTest,              // Test type for verifying NMT
  mtTracing,           // memory used for Tracing
  mtLogging,           // memory for logging
  mtArguments,         // memory for argument processing
  mtModule,            // memory for module processing
  mtSynchronizer,      // memory for synchronization primitives
  mtSafepoint,         // memory for safepoint support
  mtNone,              // undefined
  mt_number_of_types   // number of memory types (mtDontTrack
                       // is not included as validate type)
};
```
* CHeapObj和StackObj
  这个实质对应了进程的堆区和栈区，在CHeapObj中重载了new和delete方法，下面的对象在分配对象的时候实际直接通过调用系统的malloc进行分配，但是CHeapObj重载了placement new的方式可以不需要重新申请内存即可达成分配。
* Arena
  Arena负责小块内存的分配
* MetaspaceObj
  Metaspace区域的内存分配，在分配的过程中调用了MetaspaceArena来进行单独的分配。MetaSpace空间通过一个双向链表来维护内存，MetaSpace空间将内存分成了一个一个metaChunk,通过level将其分成了不同大小的列表组织进行分配。
  ```c++
  static const chunklevel_t CHUNK_LEVEL_16M =    ROOT_CHUNK_LEVEL;
  static const chunklevel_t CHUNK_LEVEL_8M =    (ROOT_CHUNK_LEVEL + 1);
  static const chunklevel_t CHUNK_LEVEL_4M =    (ROOT_CHUNK_LEVEL + 2);
  static const chunklevel_t CHUNK_LEVEL_2M =    (ROOT_CHUNK_LEVEL + 3);
  static const chunklevel_t CHUNK_LEVEL_1M =    (ROOT_CHUNK_LEVEL + 4);
  static const chunklevel_t CHUNK_LEVEL_512K =  (ROOT_CHUNK_LEVEL + 5);
  static const chunklevel_t CHUNK_LEVEL_256K =  (ROOT_CHUNK_LEVEL + 6);
  static const chunklevel_t CHUNK_LEVEL_128K =  (ROOT_CHUNK_LEVEL + 7);
  static const chunklevel_t CHUNK_LEVEL_64K =   (ROOT_CHUNK_LEVEL + 8);
  static const chunklevel_t CHUNK_LEVEL_32K =   (ROOT_CHUNK_LEVEL + 9);
  static const chunklevel_t CHUNK_LEVEL_16K =   (ROOT_CHUNK_LEVEL + 10);
  static const chunklevel_t CHUNK_LEVEL_8K =    (ROOT_CHUNK_LEVEL + 11);
  static const chunklevel_t CHUNK_LEVEL_4K =    (ROOT_CHUNK_LEVEL + 12);
  static const chunklevel_t CHUNK_LEVEL_2K =    (ROOT_CHUNK_LEVEL + 13);
  static const chunklevel_t CHUNK_LEVEL_1K =    (ROOT_CHUNK_LEVEL + 14);
  ```
* ResourceObj