---
title: jvm tune param
date: 2023-12-20 03:17:50
categories: jvm
tags: param
---
## 调优参数

### gc 参数
* -XX:GCTimeRatio
* -XX:G1HeapRegionSize
* -XX:InitiatingHeapOccupancyPercent
* -XX:G1MixedGCCountTarget
* -XX:G1HeapWastePercent
* -XX:G1OldCSetRegionThresholdPercent
* -XX:ConcGCThreads
* -XX:G1MaxNewSizePercent
* -XX:+G1UseAdaptiveIHOP
* -XX:G1MixedGCLiveThresholdPercent
* -XX:NewRatio
* -XX:SurvivorRatio
* -XX:MaxTenuringThreshold
* -XX:ParallelGCThreads
* -XX:G1HeapResizePolicy
* -XX:PLABWeight
### 内存参数
* -XX:MallocExtConf=tcache:true,dirty_decay_ms:10000,muzzy_decay_ms:10000,narenas:10
* -XX:-PreserveFramePointer
### 预取参数
* -XX:+AlwaysPreTouch 
* -XX:PrefetchCopyIntervalInBytes
* -XX:PrefetchScanIntervalInBytes
* -XX:PrefetchFieldsAhead
### JIT参数
* -XX:+EnableJVMCI
* -XX:+UseJVMCICompiler

* -Dgraal.TrivialInliningSize
* -Dgraal.MaximumInliningSize
* -Dgraal.SmallCompiledLowLevelGraphSize
* -XX:MaxTrivialSize
* -XX:MaxInlineSize
* -XX:InlineSmallCode
### THP
* -XX:+UseCodeAreaLargePages
* -XX:+UseLargePages 
* -XX:+UseTransparentHugePages
### 协程
* -XX:+UseWisp2