package com.andy.ui;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

/**
 * @Description:DL UI测试案列
 * @Author: Andy Hoo
 * @Date: 2017/10/2 17:45
 */
public class UIServerExamples {
    public static void main(String[] args) {
        //初始化用户界面后端
        UIServer uiServer = UIServer.getInstance();

        //设置网络信息（随时间变化的梯度、分值等）的存储位置。这里将其存储于内存。
        StatsStorage statsStorage = new InMemoryStatsStorage();         //或者： new FileStatsStorage(File)，用于后续的保存和载入

        //将StatsStorage实例连接至用户界面，让StatsStorage的内容能够被可视化
        uiServer.attach(statsStorage);
    }
}
