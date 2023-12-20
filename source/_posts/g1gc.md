---
title: g1gc
date: 2023-12-20 03:19:27
categories: jvm
tags: gc
---
## G1 GC算法
### 内存模型
#### 青年代
* Eden
初始分配在Eden区，在出发young GC的时候会将存活对象晋升到Survivor中的一个分区
* Survivor
等分为两个部分，简称1和2，Survivor1区存放Young GC中Eden区存活的数据，Survivor2存放YoungGC中Survivor1中存活的数据，通过设置JVM参数-XX:MaxTenuringThreshold可以调整比例。-XX:MaxTenuringThreshold代表晋升年龄最大阈值，默认15。在新生代中对象存活次数(经过YGC的次数)后仍然存活，就会晋升到老年代。每经过一次YGC，年龄加1，当survivor区的对象年龄达到TenuringThreshold时，表示该对象是长存活对象，就会直接晋升到老年代。

整个年轻代内存会在初始空间 -XX:G1NewSizePercent（默认整堆5%）与最大空间 -XX:G1MaxNewSizePercent（默认60%）之间动态变化，且由参数目标暂停时间 -XX:MaxGCPauseMillis（默认200ms）、需要扩缩容的大小以及分区的已记忆集合（RSet）计算得到。当然，G1 依然可以设置固定的年轻代大小（参数 -XX:NewRatio、-Xmn），但同时暂停目标将失去意义。
* Eden和Survivor占比
默认情况下Eden:Survivor1:Survivor2 为 8:1:1的比例
-XX:TargetSurvivorRatio
  设定survivor区的目标使用率。默认50，即survivor区对象目标使用率为50%。

  JVM会将每个对象的年龄信息、各个年龄段对象的总大小记录在“age table”表中。基于“age table”、survivor区大小、survivor区目标使用率（-XX:TargetSurvivorRatio）、晋升年龄阈值（-XX:MaxTenuringThreshold），JVM会动态的计算tenuring threshold的值。一旦对象年龄达到了tenuring threshold就会晋升到老年代。

  为什么要动态的计算tenuring threshold的值呢？假设有很多年龄还未达到TenuringThreshold的对象依旧停留在survivor区，这样不利于新对象从eden晋升到survivor。因此设置survivor区的目标使用率，当使用率达到时重新调整TenuringThreshold值，让对象尽早的去old区。
#### 老年代
#### 整体内存和GC内存分配
* 分区 Region
G1 采用了分区（Region）的思路，将整个堆空间分成若干个大小相等的内存区域，每次分配对象空间将逐段地使用内存。因此，在堆的使用上，G1 并不要求对象的存储一定是物理上连续的，只要逻辑上连续即可；每个分区也不会确定地为某个代服务，可以按需在年轻代和老年代之间切换。启动时可以通过参数 -XX:G1HeapRegionSize=n 可指定分区大小（1MB~32MB，且必须是2的幂），默认将整堆划分为2048个分区。
* 卡片 Card
在每个分区内部又被分成了若干个大小为512 Byte 卡片（Card），标识堆内存最小可用粒度所有分区的卡片将会记录在全局卡片表（Global Card Table）中，分配的对象会占用物理上连续的若干个卡片，当查找对分区内对象的引用时便可通过记录卡片来查找该引用对象（见 RSet）。每次对内存的回收，都是对指定分区的卡片进行处理。
* 堆 Heap
G1 同样可以通过 -Xms/-Xmx 来指定堆空间大小。当发生年轻代收集或混合收集时，通过计算 GC 与应用的耗费时间比，自动调整堆空间大小。如果 GC 频率太高，则通过增加堆尺寸，来减少 GC 频率，相应地 GC 占用的时间也随之降低；目标参数 -XX:GCTimeRatio 即为 GC 与应用的耗费时间比，G1 默认为 9，而 CMS 默认为 99，因为 CMS 的设计原则是耗费在 GC 上的时间尽可能的少。另外，当空间不足，如对象空间分配或转移失败时，G1 会首先尝试增加堆空间，如果扩容失败，则发起担保的 Full GC。Full GC 后，堆尺寸计算结果也会调整堆空间。
### 垃圾回收类别
* Young GC
* MixedGC
* FullGC
### 垃圾回收阶段
* 初始标记
* 根扫描
* 并发标记
* 存活数据计算
* 重新标记
* 清除

参考：http://www.linkedkeeper.com/1511.html