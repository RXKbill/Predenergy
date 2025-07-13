// 设备数据
const deviceData = {
    solar: [
        { name: "成都光伏电站", value: [104.06, 30.67, 200], capacity: "200MW", status: "正常" },
        { name: "绵阳光伏基地", value: [104.75, 31.47, 150], capacity: "150MW", status: "正常" },
        { name: "乐山光伏园区", value: [103.76, 29.55, 180], capacity: "180MW", status: "正常" }
    ],
    wind: [
        { name: "雅安风电场", value: [103.04, 30.01, 100], capacity: "100MW", status: "正常" },
        { name: "甘孜风电基地", value: [100.00, 31.62, 120], capacity: "120MW", status: "正常" },
        { name: "阿坝风电园区", value: [102.22, 31.89, 90], capacity: "90MW", status: "正常" }
    ]
};

let chart = null;

function initDeviceMap() {
    try {
        console.log('开始初始化设备分布地图...');
        
        const container = document.getElementById('device-map');
        if (!container) {
            console.error('找不到地图容器 #device-map');
            return;
        }

        // 设置容器尺寸
        container.style.width = '100%';
        container.style.height = '400px';
        container.style.backgroundColor = '#404a59';

        // 确保容器可见
        console.log('容器尺寸:', {
            width: container.clientWidth,
            height: container.clientHeight,
            display: window.getComputedStyle(container).display
        });

        // 销毁现有实例
        if (chart) {
            chart.dispose();
        }

        // 创建新实例
        chart = echarts.init(container);
        console.log('ECharts 实例创建成功');

        const option = {
            backgroundColor: '#404a59',
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    if (params.data) {
                        return `
                            <div style="padding: 5px">
                                <h6 style="margin: 0 0 5px">${params.data.name}</h6>
                                <p style="margin: 0">装机容量: ${params.data.capacity}</p>
                                <p style="margin: 0">运行状态: ${params.data.status}</p>
                            </div>
                        `;
                    }
                    return params.name;
                }
            },
            visualMap: {
                min: 0,
                max: 200,
                calculable: true,
                inRange: {
                    color: ['#50a3ba', '#eac736', '#d94e5d']
                },
                textStyle: {
                    color: '#fff'
                }
            },
            geo: {
                map: 'china',
                roam: true,
                zoom: 1.8,
                center: [104.06, 30.67],
                label: {
                    emphasis: {
                        show: true,
                        color: '#fff'
                    }
                },
                itemStyle: {
                    normal: {
                        areaColor: '#323c48',
                        borderColor: '#111'
                    },
                    emphasis: {
                        areaColor: '#2a333d'
                    }
                }
            },
            series: [
                {
                    name: '光伏电站',
                    type: 'scatter',
                    coordinateSystem: 'geo',
                    data: deviceData.solar,
                    symbolSize: function (val) {
                        return val[2] / 10;
                    },
                    label: {
                        formatter: '{b}',
                        position: 'right',
                        show: false
                    },
                    itemStyle: {
                        color: '#ffd700'
                    },
                    emphasis: {
                        label: {
                            show: true
                        }
                    }
                },
                {
                    name: '风电场',
                    type: 'scatter',
                    coordinateSystem: 'geo',
                    data: deviceData.wind,
                    symbolSize: function (val) {
                        return val[2] / 10;
                    },
                    label: {
                        formatter: '{b}',
                        position: 'right',
                        show: false
                    },
                    itemStyle: {
                        color: '#4169e1'
                    },
                    emphasis: {
                        label: {
                            show: true
                        }
                    }
                }
            ]
        };

        // 应用配置
        chart.setOption(option);
        console.log('地图配置已应用');

        // 添加按钮事件监听
        const buttons = {
            showSolar: document.getElementById('showSolar'),
            showWind: document.getElementById('showWind'),
            showAll: document.getElementById('showAll')
        };

        if (buttons.showSolar && buttons.showWind && buttons.showAll) {
            buttons.showSolar.addEventListener('click', () => {
                option.series[0].show = true;
                option.series[1].show = false;
                chart.setOption(option);
            });

            buttons.showWind.addEventListener('click', () => {
                option.series[0].show = false;
                option.series[1].show = true;
                chart.setOption(option);
            });

            buttons.showAll.addEventListener('click', () => {
                option.series[0].show = true;
                option.series[1].show = true;
                chart.setOption(option);
            });
        }

        // 监听窗口大小变化
        window.addEventListener('resize', () => {
            if (chart && container.clientWidth > 0 && container.clientHeight > 0) {
                chart.resize();
            }
        });

    } catch (error) {
        console.error('地图初始化失败:', error);
    }
}

// 确保在页面和资源加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM加载完成，检查依赖...');
    
    // 检查必要的依赖
    if (typeof echarts === 'undefined') {
        console.error('ECharts 未加载');
        return;
    }

    if (!echarts.getMap('china')) {
        console.error('中国地图数据未加载');
        return;
    }

    // 延迟初始化以确保DOM完全准备好
    setTimeout(() => {
        console.log('开始延迟初始化...');
        initDeviceMap();
    }, 500);
}); 