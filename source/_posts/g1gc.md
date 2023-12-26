---
title: g1gc
date: 2023-12-20 03:19:27
categories: jvm
tags: gc
---
# G1 GC算法
G1GC的本质是一个标记复制算法。根据《深入理解Java虚拟机：JVM高级特性与最佳实践（第3版）周志明》中介绍的垃圾回收算法为例，基本的垃圾回收算法可能包括标记清除，标记复制，标记整理算法，其中G1GC属于标记复制算法
## 内存模型
### **分代划分**
* **青年代**
青年代表示刚刚分配内存的区域，首先Java应用线程在分配的时候会优先在青年代进行分配，G1算法只有在不断Young gc的过程中慢慢的填满老年代。在达到IHOP的设定值的时候才会触发MixedGC
    * Eden
    初始分配在Eden区，在出发young GC的时候会将存活对象晋升到Survivor中的一个分区
    * Survivor
    等分为两个部分，简称1和2，Survivor1区存放Young GC中Eden区存活的数据，Survivor2存放YoungGC中Survivor1中存活的数据，通过设置JVM参数-XX:MaxTenuringThreshold可以调整比例。-XX:MaxTenuringThreshold代表晋升年龄最大阈值，默认15。在新生代中对象存活次数(经过YGC的次数)后仍然存活，就会晋升到老年代。每经过一次YGC，年龄加1，当survivor区的对象年龄达到TenuringThreshold时，表示该对象是长存活对象，就会直接晋升到老年代。

    整个年轻代内存会在初始空间 -XX:G1NewSizePercent（默认整堆5%）与最大空间 -XX:G1MaxNewSizePercent（默认60%）之间动态变化，且由参数目标暂停时间 -XX:MaxGCPauseMillis（默认200ms）、需要扩缩容的大小以及分区的已记忆集合（RSet）计算得到。当然，G1 依然可以设置固定的年轻代大小（参数 -XX:NewRatio、-Xmn），但同时暂停目标将失去意义。
    * Eden和Survivor占比
    默认情况下Eden:Survivor1:Survivor2 为 8:1:1的比例
        * -XX:TargetSurvivorRatio

            设定survivor区的目标使用率。默认50，即survivor区对象目标使用率为50%。参数代表着动态调整tenuringThreshold的值，如果survivior区中存活率过大则会减少晋升阈值，反之则会增加。
            
            JVM会将每个对象的年龄信息、各个年龄段对象的总大小记录在“age table”表中。基于“age table”、survivor区大小、survivor区目标使用率（-XX:TargetSurvivorRatio）、晋升年龄阈值（-XX:MaxTenuringThreshold），JVM会动态的计算tenuring threshold的值。一旦对象年龄达到了tenuring threshold就会晋升到老年代。
            
            为什么要动态的计算tenuring threshold的值呢？假设有很多年龄还未达到TenuringThreshold的对象依旧停留在survivor区，这样不利于新对象从eden晋升到survivor。因此设置survivor区的目标使用率，当使用率达到时重新调整TenuringThreshold值，让对象尽早的去old区。
* **老年代**

老年代负责存放长期存活的对象，老年代的对象是经历过多次G1gc之后仍然存活的对象。

### **整体内存和GC内存分配**
* 分区 Region
G1 采用了分区（Region）的思路，将整个堆空间分成若干个大小相等的内存区域，每次分配对象空间将逐段地使用内存。因此，在堆的使用上，G1 并不要求对象的存储一定是物理上连续的，只要逻辑上连续即可；每个分区也不会确定地为某个代服务，可以按需在年轻代和老年代之间切换。启动时可以通过参数 -XX:G1HeapRegionSize=n 可指定分区大小（1MB~32MB，且必须是2的幂），默认将整堆划分为2048个分区。
* 卡片 Card
在每个分区内部又被分成了若干个大小为512 Byte 卡片（Card），标识堆内存最小可用粒度所有分区的卡片将会记录在全局卡片表（Global Card Table）中，分配的对象会占用物理上连续的若干个卡片，当查找对分区内对象的引用时便可通过记录卡片来查找该引用对象（见 RSet）。每次对内存的回收，都是对指定分区的卡片进行处理。
* 堆 Heap
G1 同样可以通过 -Xms/-Xmx 来指定堆空间大小。当发生年轻代收集或混合收集时，通过计算 GC 与应用的耗费时间比，自动调整堆空间大小。如果 GC 频率太高，则通过增加堆尺寸，来减少 GC 频率，相应地 GC 占用的时间也随之降低；目标参数 -XX:GCTimeRatio 即为 GC 与应用的耗费时间比，G1 默认为 9，而 CMS 默认为 99，因为 CMS 的设计原则是耗费在 GC 上的时间尽可能的少。另外，当空间不足，如对象空间分配或转移失败时，G1 会首先尝试增加堆空间，如果扩容失败，则发起担保的 Full GC。Full GC 后，堆尺寸计算结果也会调整堆空间。
## **垃圾回收类别**
* Young GC
年轻代的回收算法，Eden区进行扫描，是一个STW的过程，在YoungGC中存活的对象分为如下两种情况。如果数据在Eden区那么存活对象会将其放在Survivor1区，如果Survivor1区中存活的对象晋升到Survivor2区，根据Teuring判断是否需要晋升到老年代
    * **TLAB(Thread Local Allocation Buffer)**:线程本地缓存，TLAB分配在Eden区上，对于每一个本地想成有单独的TLAB，每一个TLAB不会过大。TLAB为了解决线程分配的冲突问题，对于每一个Java应用线程内存可以单独分配，不需要对于堆进行冲突访问。因为堆区是一块Java应用线程共享的内存空间，而且是一个同步访问的过程。因此开启TLAB优化可以大幅加快分配速度。可以了解逃逸分析相关问题，类似于栈上分配。需要注意的是TLAB空间有限，对于本地线程有单独的Buffer进行分配，只有在初始的分配阶段是独占的，在后续使用的过程中也可以进行跨线程的访问。Buffer空间有限每个线程对应一个Buffer空间，如果对象过大无法在TLAB上面进行分配那么则需要在堆空间进行分配。
    * **PLAB(Promote Local Allocation Buffer)**:对于进程对象的分配区域，主要存在survivor区。其目的也是为了解决共享堆的分配冲突问题，PLAB负责的是晋升过程的加速分配，因此主要存在于survivior区。
* MixedGC

    本质是对于Cset中的数据进行回收，功能上类似于YoungGC只不过Cset中存在需要Old区的数据。Cset会不断的启发式的加入需要回收的数据。
* FullGC

## 垃圾回收阶段
* **Young GC**
    1. 将eden区和survivor区所有Region添加到CSet中准备回收。注：CSet是一个存放待回收Region的数据结构。

    2. 标记阶段
    
        从GC Roots开始标记直接引用的新生代对象。基于RSet标记跨代引用的新生代对象。
    3. 复制阶段：将标记活跃的对象复制到其他Region中。
* **并发标记周期**
    * 初始标记
    
        这个过程需要进入Stop the World的，仅仅只是标记一下GC Roots直接能引用的对象，这个过程速度是很快的。先停止系统程序的运行，然后对各个线程栈内存中的局部变量代表的GC Roots，以及方法区中的类静态变量代表的GC Roots，进行扫描，标记出来他们直接引用的那些对象

        GC Root中数据为活跃数据可能包括
        1. JavaStack中的引用的对象。
        2. 方法区中静态引用指向的对象。
        3. 方法区中常量引用指向的对象。
        4. Native方法中JNI引用的对象。
    * 并发标记

        这个阶段会允许系统程序的运行，同时进行GC Roots追踪，从GC Roots开始追踪所有的存活对象，并对这个过程对象的变化做记录，比如哪些对象失去了引用，哪些对象是新建的。这个阶段也是很耗时的，要追踪全部存活的对象，但跟系统并发运行，影响不大。
    * 重新标记

        这个阶段会进入Stop the World，系统程序是禁止运行的，但是会根据并发标记阶段记录的那些对象修改，最终标记一下有哪些存活对象，有哪些是垃圾对象。
    * 清除

        计算存活对象数量：在并发标记阶段，垃圾收集器会遍历对象图并标记存活对象。这个过程可以帮助G1GC了解每个区域的存活对象数量。
        计算回收收益：根据每个区域的存活对象数量，G1GC会计算回收收益，即回收某个区域可以释放多少空间。
        选择回收集合：在计算完回收收益后，G1GC会根据预设的暂停时间目标（例如通过-XX:MaxGCPauseMillis参数设置）来选择哪些区域应该被包含在回收集合中。这个过程中，G1GC会优先选择回收收益较高的区域。

        这个阶段G1允许多次执行混合回收，也就是说先停止系统工作，执行回收，恢复系统运行，再停止系统运行，再回收，再恢复…这么一个流程。每次回收的间隔是由G1自己控制的，回收执行次数可以通过参数-XX:G1MixedGCCountTarget来设置，这个参数默认回收次数是8次，同时有一个参数-XX:G1HeapWastePercent，默认值是整个堆大小5%，就是说当前回收集合内即将空出来的区域大于整个堆的5%，就会立即停止混合回收了。正常默认回收次数是8次，但是可能到了4次，发现空闲Region大于整个堆的5%，就不会再进行后续回收了。
* **Mixed GC**

    MixedGC的本质和YoungGC没有区别，都是对于Cset中的数据进行垃圾回收，但是不同的点在于Old区中的数据同样会加入到Cset中进行回收。

参考：http://www.linkedkeeper.com/1511.html