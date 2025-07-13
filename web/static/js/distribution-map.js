// 全局变量保存chart实例
let chart = null;

// 状态更新函数
function updateStatus(statusId, message, isSuccess) {
    const element = document.getElementById(statusId);
    if (element) {
        element.textContent = message;
        element.style.color = isSuccess ? '#4caf50' : '#ff5252';
    }
}

// 初始化函数
function initMap() {
    console.log('开始初始化地图...');
    
    // 检查依赖
    if (!window.echarts) {
        console.error('ECharts 未加载');
        return;
    }

    if (!echarts.getMap('china')) {
        console.error('中国地图数据未加载');
        return;
    }

    const container = document.getElementById('distribution-map');
    if (!container) {
        console.error('找不到地图容器');
        return;
    }

    // 检查容器尺寸
    const { width, height } = container.getBoundingClientRect();
    console.log('容器尺寸:', width, height);
    
    if (width === 0 || height === 0) {
        console.warn('容器尺寸为0，等待重试...');
        setTimeout(initMap, 500);
        return;
    }

    // 如果已存在实例，先销毁
    if (chart) {
        chart.dispose();
    }

    // 创建地图实例
    try {
        chart = echarts.init(container);
        console.log('地图实例创建成功');

        // 基础数据
        const deviceData = {
            solar: [
                { name: "成都光伏电站", value: [104.06, 30.67, 200], capacity: "200MW" },
                { name: "绵阳光伏基地", value: [104.75, 31.47, 150], capacity: "150MW" },
                { name: "乐山光伏园区", value: [103.76, 29.55, 180], capacity: "180MW" }
            ],
            wind: [
                { name: "雅安风电场", value: [103.04, 30.01, 100], capacity: "100MW" },
                { name: "甘孜风电基地", value: [100.00, 31.62, 120], capacity: "120MW" },
                { name: "阿坝风电园区", value: [102.22, 31.89, 90], capacity: "90MW" }
            ]
        };

        // 配置选项
        const option = {
            backgroundColor: '#404a59',
            tooltip: {
                trigger: 'item',
                formatter: '{b}<br/>装机容量: {c}MW'
            },
            legend: {
                orient: 'vertical',
                right: 10,
                top: 'center',
                data: ['光伏电站', '风电场'],
                textStyle: { color: '#fff' }
            },
            geo: {
                map: 'china',
                roam: true,
                zoom: 4,
                center: [104.06, 30.67],
                label: {
                    show: true,
                    color: '#fff'
                },
                itemStyle: {
                    areaColor: '#323c48',
                    borderColor: '#111'
                },
                emphasis: {
                    itemStyle: {
                        areaColor: '#2a333d'
                    }
                }
            },
            series: [
                {
                    name: '光伏电站',
                    type: 'effectScatter',
                    coordinateSystem: 'geo',
                    data: deviceData.solar,
                    symbolSize: 15,
                    itemStyle: { color: '#ffd700' },
                    rippleEffect: {
                        scale: 3,
                        brushType: 'stroke'
                    }
                },
                {
                    name: '风电场',
                    type: 'effectScatter',
                    coordinateSystem: 'geo',
                    data: deviceData.wind,
                    symbolSize: 15,
                    itemStyle: { color: '#4169e1' },
                    rippleEffect: {
                        scale: 3,
                        brushType: 'stroke'
                    }
                }
            ]
        };

        // 应用配置
        chart.setOption(option);
        console.log('地图配置已应用');

        // 绑定按钮事件
        const solarBtn = document.getElementById('showSolarBtn');
        const windBtn = document.getElementById('showWindBtn');
        const allBtn = document.getElementById('showAllBtn');

        if (solarBtn) {
            solarBtn.onclick = function() {
                console.log('显示光伏电站');
                chart.setOption({
                    series: [
                        { name: '光伏电站', show: true },
                        { name: '风电场', show: false }
                    ]
                });
            };
        }

        if (windBtn) {
            windBtn.onclick = function() {
                console.log('显示风电场');
                chart.setOption({
                    series: [
                        { name: '光伏电站', show: false },
                        { name: '风电场', show: true }
                    ]
                });
            };
        }

        if (allBtn) {
            allBtn.onclick = function() {
                console.log('显示所有设备');
                chart.setOption({
                    series: [
                        { name: '光伏电站', show: true },
                        { name: '风电场', show: true }
                    ]
                });
            };
        }

        // 响应窗口大小变化
        window.addEventListener('resize', function() {
            if (chart) {
                chart.resize();
            }
        });

    } catch (error) {
        console.error('初始化失败:', error);
    }
}

// 确保在DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM加载完成，初始化地图...');
    setTimeout(initMap, 100);
});

// 暴露全局测试函数
window.testMap = initMap; 