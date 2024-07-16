---
title: cpp-并发一致性
date: 2024-01-26 12:47:49
tags:
---
* cache 一致性
    * register buffer/cache: ~1 cycle < 1ns
    * L1 cache: ~3 cycles ~ 1ns, 单核心独占
    * L2 cache: ~12 cycles ~ 3ns, 双核心共享
    * L3 cache: ~38 cycles ~ 12ns
    * DRAM: ~65ns
    * MESI协议: MESI是一种维护cacheline状态一致性的协议。MESI: 由Modified,Exclusive,Share,Invalid四种状态组合。
        * Modified(M):
            当一个core 的cacheline的状态是M时，说明当前core最近修改了这个cache，那么其他core的cache不能再修改当前cache line对应的memory location，除非该cache将这个修改同步到了memory。这个core对这个memory location可以理解为Owned。
        * Exclusive(E):
            E这个状态与M很像，区别在于当前core并没有修改当前的cacheline，这意味着当前cacheline存储的memory location的值是最新的。当前core可以对该cacheline进行modify且不需要与其他core的cache同步。这个core对这个memory location可以理解为Owned。
        * Share(S):
            S表示当前cacheline在其他core的cache也存在copy，当前core如果需要修改该cacheline则需要与其他core的cache进行提前沟通。
        * Invalid(I):
            I表示当前cacheline是空的。

    * protocol message分为以下几种消息类型：
        * Read
        当一个cache需要读取某个cacheline消息的时候就会发起read消息。
        * Read Response
        read response是read的回应，这response可以来自其他core的cache也可以来自memory。当其他core中对当前cacheline是M状态时，则会发起read response。
        * Invalidate
        Invalidate消息包含对应的memory location，接收到这个消息的cache需要将自己cacheline内容剔除，并响应。
        Invalidate acknowledge
        接收到Invalidate后删除cacheline中的数据就向发起者回复invalidate ack。
        * Read Invalidate
        这个消息包含两个操作，read和invalidate，那么它也需要接收read response和多个invalidate ack响应。
        * Write Back
        writeback包含数据和地址，会将这个地址对应的数据刷到内存中。
* 关系术语
    * Sequenced-before：必须是完全的顺序，同一个线程语句A执行在B之前，此时满足sequenced-before关系
    * Happens-before：happens-before关系表示的不同线程之间的操作先后顺序。如果A happens-before B，则A的内存状态将在B操作执行之前就可见。
    * Synchronizes-with：synchronizes-with关系强调的是变量被修改之后的传播关系（propagate）， 即如果一个线程修改某变量的之后的结果能被其它线程可见，那么就是满足synchronizes-with关系的
    * Carries dependency：同一个线程内，表达式A sequenced-before 表达式B，并且表达式B的值是受表达式A的影响的一种关系， 称之为"Carries dependency"。
* c++ 内存一致性
    * memory_order_relaxed：不对顺序做任何保证
    * memory_order_consume：本线程中有关本原子类型的操作，必须在本次原子操作之后
    * memory_order_acquire：本线程中所有读操作，必须在本次原子操作之后
    * memory_order_release：本线程中所有写操作完成了，才可以开始本次原子操作
    * memory_order_acq_rel：同时满足acquire和release
    * memory_order_seq_cst：完全顺序化执行