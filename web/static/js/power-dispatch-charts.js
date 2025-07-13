// 电力调度专业图表实现

class PowerDispatchCharts {
    constructor(options = {}) {
        // 防止重复初始化
        if (window.powerDispatchChartsInstance) {
            return window.powerDispatchChartsInstance;
        }
        window.powerDispatchChartsInstance = this;

        // 配置选项
        this.options = {
            powerCurveContainerId: options.powerCurveContainerId || 'power-curve-chart',
            powerCompositionContainerId: options.powerCompositionContainerId || 'power-composition-chart',
            theme: options.theme || 'dark'
        };

        // 绑定标签页切换事件
        this.bindTabEvents();
    }

    // 绑定标签页切换事件
    bindTabEvents() {
        document.querySelectorAll('a[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const targetId = e.target.getAttribute('href');
                if (targetId && targetId.includes('dispatch')) {
                    console.log('Dispatch tab shown, initializing charts...');
                    // 延迟初始化以确保DOM已更新
                    setTimeout(() => {
                        this.initializeCharts();
                    }, 100);
                }
            });
        });

        // 初始化时检查是否在调度标签页
        const activeTab = document.querySelector('a[data-bs-toggle="tab"].active');
        if (activeTab && activeTab.getAttribute('href').includes('dispatch')) {
            console.log('Initially on dispatch tab, initializing charts...');
            this.initializeCharts();
        }
    }

    // 初始化所有图表
    initializeCharts() {
        console.log('Initializing power dispatch charts...');
        
        // 确保 ECharts 已加载
        if (typeof echarts === 'undefined') {
            console.error('ECharts library not found. Please include echarts.min.js');
            return;
        }

        this.initializePowerCurveChart();
        this.initializePowerCompositionChart();
        
        // 启动实时数据更新
        this.startRealTimeUpdates();
    }

    // 初始化功率曲线图
    initializePowerCurveChart() {
        // 尝试多个可能的选择器
        const powerCurveElement = document.getElementById(this.options.powerCurveContainerId) || 
                                document.querySelector(`[data-chart="${this.options.powerCurveContainerId}"]`) ||
                                document.querySelector('.power-curve-chart');

        if (!powerCurveElement) {
            console.warn('Power curve chart element not found');
            return;
        }

        console.log('Found power curve element:', powerCurveElement);

        try {
            // 生成示例数据
            const timePoints = this.generateTimePoints(24); // 24小时数据
            const actualPower = this.generatePowerData(timePoints, 500, 800); // 实际功率
            const plannedPower = this.generatePowerData(timePoints, 500, 800); // 计划功率
            const availablePower = this.generatePowerData(timePoints, 600, 1000); // 可用功率
            
            const options = {
                backgroundColor: this.options.theme === 'dark' ? '#2a3042' : '#fff',
                series: [{
                    name: '实际功率',
                    type: 'line',
                    data: actualPower,
                    smooth: true,
                    lineStyle: {
                        width: 3
                    },
                    itemStyle: {
                        color: '#4F8CBE'
                    }
                }, {
                    name: '计划功率',
                    type: 'line',
                    data: plannedPower,
                    smooth: true,
                    lineStyle: {
                        width: 3,
                        type: 'dashed'
                    },
                    itemStyle: {
                        color: '#2ab57d'
                    }
                }, {
                    name: '可用功率',
                    type: 'line',
                    data: availablePower,
                    smooth: true,
                    lineStyle: {
                        width: 2,
                        type: 'dotted'
                    },
                    itemStyle: {
                        color: '#ffbf53'
                    },
                    areaStyle: {
                        color: {
                            type: 'linear',
                            x: 0,
                            y: 0,
                            x2: 0,
                            y2: 1,
                            colorStops: [{
                                offset: 0,
                                color: 'rgba(255, 191, 83, 0.2)'
                            }, {
                                offset: 1,
                                color: 'rgba(255, 191, 83, 0)'
                            }]
                        }
                    }
                }],
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        let result = params[0].axisValue + '<br/>';
                        params.forEach(param => {
                            result += param.marker + param.seriesName + ': ' + param.value + ' MW<br/>';
                        });
                        return result;
                    }
                },
                legend: {
                    data: ['实际功率', '计划功率', '可用功率'],
                    textStyle: {
                        color: this.options.theme === 'dark' ? '#ccc' : '#333'
                    }
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    boundaryGap: false,
                    data: timePoints,
                    axisLabel: {
                        color: this.options.theme === 'dark' ? '#ccc' : '#333',
                        formatter: '{value}:00'
                    },
                    axisLine: {
                        lineStyle: {
                            color: this.options.theme === 'dark' ? '#ccc' : '#333'
                        }
                    }
                },
                yAxis: {
                    type: 'value',
                    name: '功率 (MW)',
                    nameTextStyle: {
                        color: this.options.theme === 'dark' ? '#ccc' : '#333'
                    },
                    axisLabel: {
                        color: this.options.theme === 'dark' ? '#ccc' : '#333'
                    },
                    axisLine: {
                        lineStyle: {
                            color: this.options.theme === 'dark' ? '#ccc' : '#333'
                        }
                    },
                    splitLine: {
                        lineStyle: {
                            color: this.options.theme === 'dark' ? 'rgba(204, 204, 204, 0.2)' : 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                }
            };

            console.log('Initializing power curve chart with options:', options);
            this.powerCurveChart = echarts.init(powerCurveElement);
            this.powerCurveChart.setOption(options);
            console.log('Power curve chart initialized successfully');

            // 响应窗口大小变化
            window.addEventListener('resize', () => {
                this.powerCurveChart && this.powerCurveChart.resize();
            });
        } catch (error) {
            console.error('Error initializing power curve chart:', error);
        }
    }

    // 初始化电力构成图
    initializePowerCompositionChart() {
        // 尝试多个可能的选择器
        const powerCompositionElement = document.getElementById(this.options.powerCompositionContainerId) || 
                                      document.querySelector(`[data-chart="${this.options.powerCompositionContainerId}"]`) ||
                                      document.querySelector('.power-composition-chart');

        if (!powerCompositionElement) {
            console.warn('Power composition chart element not found');
            return;
        }

        console.log('Found power composition element:', powerCompositionElement);

        try {
            const options = {
                backgroundColor: this.options.theme === 'dark' ? '#2a3042' : '#fff',
                tooltip: {
                    trigger: 'item',
                    formatter: '{a} <br/>{b}: {c} MW ({d}%)'
                },
                legend: {
                    orient: 'vertical',
                    right: 10,
                    top: 'center',
                    textStyle: {
                        color: this.options.theme === 'dark' ? '#ccc' : '#333'
                    },
                    data: ['火电', '水电', '风电', '光伏', '储能']
                },
                series: [
                    {
                        name: '电力构成',
                        type: 'pie',
                        radius: ['40%', '70%'],
                        avoidLabelOverlap: false,
                        itemStyle: {
                            borderRadius: 10,
                            borderColor: this.options.theme === 'dark' ? '#2a3042' : '#fff',
                            borderWidth: 2
                        },
                        label: {
                            show: false,
                            position: 'center'
                        },
                        emphasis: {
                            label: {
                                show: true,
                                fontSize: '14',
                                fontWeight: 'bold',
                                color: this.options.theme === 'dark' ? '#ccc' : '#333'
                            }
                        },
                        labelLine: {
                            show: false
                        },
                        data: [
                            { value: 450, name: '火电', itemStyle: { color: '#ef5350' } },
                            { value: 300, name: '水电', itemStyle: { color: '#4F8CBE' } },
                            { value: 200, name: '风电', itemStyle: { color: '#2ab57d' } },
                            { value: 150, name: '光伏', itemStyle: { color: '#ffbf53' } },
                            { value: 100, name: '储能', itemStyle: { color: '#556ee6' } }
                        ]
                    }
                ]
            };

            console.log('Initializing power composition chart with options:', options);
            this.powerCompositionChart = echarts.init(powerCompositionElement);
            this.powerCompositionChart.setOption(options);
            console.log('Power composition chart initialized successfully');

            // 响应窗口大小变化
            window.addEventListener('resize', () => {
                this.powerCompositionChart && this.powerCompositionChart.resize();
            });
        } catch (error) {
            console.error('Error initializing power composition chart:', error);
        }
    }

    // 生成时间点数据
    generateTimePoints(hours) {
        return Array.from({length: hours}, (_, i) => i);
    }

    // 生成功率数据
    generatePowerData(timePoints, baseValue, variance) {
        return timePoints.map(hour => {
            // 模拟日间负荷变化：早晚低谷，白天高峰
            const timePattern = Math.sin((hour - 6) * Math.PI / 12) * 0.5 + 0.5;
            const randomFactor = Math.random() * 0.2 + 0.9;
            return Math.round((baseValue + variance * timePattern) * randomFactor);
        });
    }

    // 启动实时数据更新
    startRealTimeUpdates() {
        setInterval(() => {
            if (this.powerCurveChart) {
                const data = this.powerCurveChart.getOption().series;
                // 更新实际功率数据
                const newValue = Math.round((Math.random() * 200 + 500));
                data[0].data.shift();
                data[0].data.push(newValue);
                this.powerCurveChart.setOption({
                    series: data
                });
            }

            if (this.powerCompositionChart) {
                // 随机调整电力构成
                const data = this.powerCompositionChart.getOption().series[0].data;
                data.forEach(item => {
                    item.value = Math.round(item.value * (0.95 + Math.random() * 0.1));
                });
                this.powerCompositionChart.setOption({
                    series: [{
                        data: data
                    }]
                });
            }
        }, 5000); // 每5秒更新一次
    }
}

// 当DOM加载完成后初始化图表
document.addEventListener('DOMContentLoaded', function() {
    // 检查是否已加载 ECharts
    if (typeof echarts === 'undefined') {
        console.error('ECharts library not found. Please include echarts.min.js');
        return;
    }

    // 初始化图表
    window.powerDispatchCharts = new PowerDispatchCharts({
        theme: 'dark'
    });
}); 