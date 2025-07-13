// 全局变量保存chart实例
let heatmapChart = null;
let monitorGrid = null;

// 基础数据
const stationData = {
    regions: ['高新区', '天府新区', '武侯区', '金牛区', '成华区', '锦江区', '青羊区', '双流区'],
    stations: [
        { name: "成都高新区A站", region: "高新区", type: "快充站", status: "正常", power: "120kW", ports: "10/12", usage: 83, fastPorts: 8, slowPorts: 4, faultPorts: 0, todayCharges: 156, todayEnergy: 1820.5 },
        { name: "成都高新区B站", region: "高新区", type: "快充站", status: "正常", power: "150kW", ports: "12/15", usage: 92, fastPorts: 10, slowPorts: 5, faultPorts: 0, todayCharges: 188, todayEnergy: 2250.8 },
        { name: "成都高新区C站", region: "高新区", type: "慢充站", status: "正常", power: "60kW", ports: "15/15", usage: 75, fastPorts: 0, slowPorts: 15, faultPorts: 0, todayCharges: 95, todayEnergy: 580.2 },
        { name: "天府新区A站", region: "天府新区", type: "快充站", status: "正常", power: "90kW", ports: "8/10", usage: 80, fastPorts: 6, slowPorts: 4, faultPorts: 0, todayCharges: 142, todayEnergy: 1650.8 },
        { name: "天府新区B站", region: "天府新区", type: "快充站", status: "正常", power: "180kW", ports: "16/20", usage: 88, fastPorts: 14, slowPorts: 6, faultPorts: 0, todayCharges: 210, todayEnergy: 2580.5 },
        { name: "武侯区A站", region: "武侯区", type: "快充站", status: "维护中", power: "150kW", ports: "12/15", usage: 0, fastPorts: 10, slowPorts: 5, faultPorts: 2, todayCharges: 0, todayEnergy: 0 },
        { name: "武侯区B站", region: "武侯区", type: "慢充站", status: "正常", power: "40kW", ports: "20/20", usage: 65, fastPorts: 0, slowPorts: 20, faultPorts: 0, todayCharges: 82, todayEnergy: 328.5 },
        { name: "金牛区A站", region: "金牛区", type: "慢充站", status: "正常", power: "60kW", ports: "6/8", usage: 75, fastPorts: 0, slowPorts: 8, faultPorts: 0, todayCharges: 85, todayEnergy: 510.5 },
        { name: "金牛区B站", region: "金牛区", type: "快充站", status: "正常", power: "120kW", ports: "10/10", usage: 90, fastPorts: 8, slowPorts: 2, faultPorts: 0, todayCharges: 168, todayEnergy: 1960.3 },
        { name: "成华区A站", region: "成华区", type: "快充站", status: "正常", power: "120kW", ports: "8/10", usage: 95, fastPorts: 6, slowPorts: 4, faultPorts: 0, todayCharges: 175, todayEnergy: 2100.6 },
        { name: "成华区B站", region: "成华区", type: "慢充站", status: "正常", power: "40kW", ports: "5/6", usage: 83, fastPorts: 0, slowPorts: 6, faultPorts: 0, todayCharges: 76, todayEnergy: 456.2 },
        { name: "锦江区A站", region: "锦江区", type: "快充站", status: "正常", power: "150kW", ports: "12/12", usage: 87, fastPorts: 8, slowPorts: 4, faultPorts: 0, todayCharges: 165, todayEnergy: 1980.4 },
        { name: "锦江区B站", region: "锦江区", type: "慢充站", status: "故障", power: "60kW", ports: "0/10", usage: 0, fastPorts: 0, slowPorts: 8, faultPorts: 2, todayCharges: 0, todayEnergy: 0 },
        { name: "青羊区A站", region: "青羊区", type: "快充站", status: "正常", power: "180kW", ports: "15/15", usage: 93, fastPorts: 12, slowPorts: 3, faultPorts: 0, todayCharges: 195, todayEnergy: 2340.8 },
        { name: "青羊区B站", region: "青羊区", type: "慢充站", status: "正常", power: "40kW", ports: "12/12", usage: 70, fastPorts: 0, slowPorts: 12, faultPorts: 0, todayCharges: 88, todayEnergy: 352.6 },
        { name: "双流区A站", region: "双流区", type: "快充站", status: "正常", power: "120kW", ports: "10/10", usage: 85, fastPorts: 7, slowPorts: 3, faultPorts: 0, todayCharges: 158, todayEnergy: 1890.5 },
        { name: "双流区B站", region: "双流区", type: "快充站", status: "维护中", power: "90kW", ports: "6/8", usage: 0, fastPorts: 5, slowPorts: 3, faultPorts: 1, todayCharges: 0, todayEnergy: 0 }
    ]
};

// 初始化所有图表
function initCharts() {
    console.log('开始初始化图表...');
    
    // 检查依赖
    if (typeof echarts === 'undefined') {
        console.error('ECharts 未加载，等待重试...');
        setTimeout(initCharts, 500);
        return;
    }

    // 创建图表容器
    const container = document.getElementById('distribution-map');
    if (!container) {
        console.error('找不到图表容器，等待重试...');
        setTimeout(initCharts, 500);
        return;
    }

    // 清空容器
    container.innerHTML = '';

    // 创建子容器
    const heatmapDiv = document.createElement('div');
    heatmapDiv.id = 'heatmap-chart';
    container.appendChild(heatmapDiv);

    const monitorDiv = document.createElement('div');
    monitorDiv.id = 'monitor-grid';
    container.appendChild(monitorDiv);

    // 初始化图表
    try {
        // 等待DOM更新
        setTimeout(() => {
            initHeatmapChart();
            initMonitorGrid();

            // 响应窗口大小变化
            window.addEventListener('resize', function() {
                if (heatmapChart) {
                    heatmapChart.resize();
                }
                if (monitorGrid) {
                    monitorGrid.resize();
                }
            });

            console.log('图表初始化完成');
        }, 100);
    } catch (error) {
        console.error('图表初始化失败:', error);
    }
}

// 初始化区域分布热力图
function initHeatmapChart() {
    const container = document.getElementById('heatmap-chart');
    if (!container) {
        console.error('找不到热力图容器');
        return;
    }

    if (heatmapChart) {
        heatmapChart.dispose();
    }

    heatmapChart = echarts.init(container, 'dark');

    // 统计各区域数据
    const regionData = stationData.regions.map(region => {
        const stations = stationData.stations.filter(s => s.region === region);
        const totalPorts = stations.reduce((acc, s) => acc + parseInt(s.ports.split('/')[1]), 0);
        const usedPorts = stations.reduce((acc, s) => acc + parseInt(s.ports.split('/')[0]), 0);
        const avgUsage = stations.length > 0 ? stations.reduce((acc, s) => acc + s.usage, 0) / stations.length : 0;
        const totalEnergy = stations.reduce((acc, s) => acc + s.todayEnergy, 0);
        
        return {
            region,
            stationCount: stations.length,
            totalPorts,
            usedPorts,
            usage: avgUsage,
            energy: totalEnergy,
            fastPorts: stations.reduce((acc, s) => acc + s.fastPorts, 0),
            slowPorts: stations.reduce((acc, s) => acc + s.slowPorts, 0),
            faultPorts: stations.reduce((acc, s) => acc + s.faultPorts, 0)
        };
    });

    // 计算充电量的最大值，用于设置visualMap的范围
    const maxEnergy = Math.max(...regionData.map(d => d.energy));

    const option = {
        title: {
            text: '区域分布与运营状况',
            left: 'center',
            top: 10,
            textStyle: { color: '#fff', fontSize: 16, fontWeight: 'bold' }
        },
        tooltip: {
            position: 'top',
            formatter: function (params) {
                const data = regionData.find(d => d.region === params.name);
                if (!data) return '';
                return `
                    <div style="padding: 3px;">
                        <h4 style="margin: 0 0 10px 0">${params.name}充电站概况</h4>
                        <p style="margin: 5px 0">站点数量: ${data.stationCount}个</p>
                        <p style="margin: 5px 0">充电端口: ${data.totalPorts}个</p>
                        <p style="margin: 5px 0">└ 快充端口: ${data.fastPorts}个</p>
                        <p style="margin: 5px 0">└ 慢充端口: ${data.slowPorts}个</p>
                        <p style="margin: 5px 0">└ 故障端口: ${data.faultPorts}个</p>
                        <p style="margin: 5px 0">使用率: ${data.usage.toFixed(1)}%</p>
                        <p style="margin: 5px 0">今日充电量: ${data.energy.toFixed(1)} kWh</p>
                    </div>
                `;
            }
        },
        grid: {
            left: '15%',
            right: '15%',
            top: '20%',
            bottom: '25%',  // 增加底部空间以容纳图例
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: ['端口数量', '使用率', '充电量'],
            splitArea: { show: true },
            axisLabel: { color: '#fff', fontSize: 12 }
        },
        yAxis: {
            type: 'category',
            data: stationData.regions,
            splitArea: { show: true },
            axisLabel: { color: '#fff', fontSize: 12 }
        },
        visualMap: [{
            type: 'continuous',
            min: 0,
            max: 100,
            calculable: true,
            orient: 'horizontal',
            left: '25%',  // 调整位置到左侧
            bottom: '5%',
            itemWidth: 15,  // 减小图例宽度
            itemHeight: 100,  // 设置图例高度
            textStyle: { 
                color: '#fff',
                fontSize: 10  // 减小字体大小
            },
            text: ['使用率'],  // 添加标题
            inRange: {
                color: ['#313695', '#4575b4', '#74add1', '#2c6b67', '#1f4037', 
                       '#2c5364', '#2a4858', '#cc5333', '#a83b3b']
            },
            seriesIndex: 1
        }, {
            type: 'continuous',
            min: 0,
            max: Math.ceil(maxEnergy / 1000) * 1000,
            calculable: true,
            orient: 'horizontal',
            right: '25%',  // 调整位置到右侧
            bottom: '5%',
            itemWidth: 15,  // 减小图例宽度
            itemHeight: 100,  // 设置图例高度
            textStyle: { 
                color: '#fff',
                fontSize: 10  // 减小字体大小
            },
            text: ['充电量'],  // 添加标题
            inRange: {
                color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', 
                       '#fdae61', '#f46d43', '#d73027', '#a50026']
            },
            seriesIndex: 2
        }],
        series: [{
            name: '端口数量',
            type: 'heatmap',
            data: regionData.map((d, i) => [0, i, d.totalPorts]),
            label: {
                show: true,
                formatter: function(params) {
                    return `${params.data[2]}个`;
                },
                textStyle: { 
                    color: '#fff',
                    fontSize: 12,
                    fontWeight: 'bold',
                    textShadowColor: 'rgba(0, 0, 0, 0.5)',
                    textShadowBlur: 2
                }
            }
        }, {
            name: '使用率',
            type: 'heatmap',
            data: regionData.map((d, i) => [1, i, d.usage]),
            label: {
                show: true,
                formatter: function(params) {
                    return `${params.data[2].toFixed(1)}%`;
                },
                textStyle: { 
                    color: '#fff',
                    fontSize: 12,
                    fontWeight: 'bold',
                    textShadowColor: 'rgba(0, 0, 0, 0.5)',
                    textShadowBlur: 2
                }
            }
        }, {
            name: '充电量',
            type: 'heatmap',
            data: regionData.map((d, i) => [2, i, d.energy]),
            label: {
                show: true,
                formatter: function(params) {
                    return `${params.data[2].toFixed(0)}kWh`;
                },
                textStyle: { 
                    color: '#fff',
                    fontSize: 12,
                    fontWeight: 'bold',
                    textShadowColor: 'rgba(0, 0, 0, 0.5)',
                    textShadowBlur: 2
                }
            }
        }]
    };

    heatmapChart.setOption(option);
}

// 初始化监控状态网格
function initMonitorGrid() {
    const container = document.getElementById('monitor-grid');
    if (!container) {
        console.error('找不到监控网格容器');
        return;
    }

    if (monitorGrid) {
        monitorGrid.dispose();
    }

    monitorGrid = echarts.init(container, 'dark');

    const option = {
        title: {
            text: '充电站实时监控',
            left: 'center',
            top: 10,
            textStyle: { color: '#fff', fontSize: 16, fontWeight: 'bold' }
        },
        tooltip: {
            formatter: function(params) {
                const station = stationData.stations.find(s => s.name === params.name);
                if (!station) return '';
                return `
                    <div style="padding: 3px;">
                        <h4 style="margin: 0 0 10px 0">${station.name}</h4>
                        <p style="margin: 5px 0">类型: ${station.type}</p>
                        <p style="margin: 5px 0">状态: ${station.status}</p>
                        <p style="margin: 5px 0">额定功率: ${station.power}</p>
                        <p style="margin: 5px 0">端口使用: ${station.ports}</p>
                        <p style="margin: 5px 0">└ 快充端口: ${station.fastPorts}个</p>
                        <p style="margin: 5px 0">└ 慢充端口: ${station.slowPorts}个</p>
                        <p style="margin: 5px 0">└ 故障端口: ${station.faultPorts}个</p>
                        <p style="margin: 5px 0">使用率: ${station.usage}%</p>
                        <p style="margin: 5px 0">今日充电次数: ${station.todayCharges}次</p>
                        <p style="margin: 5px 0">今日充电量: ${station.todayEnergy.toFixed(1)} kWh</p>
                    </div>
                `;
            }
        },
        series: [{
            type: 'treemap',
            width: '90%',
            height: '75%',
            top: '15%',
            roam: false,
            nodeClick: false,
            breadcrumb: { show: false },
            upperLabel: { show: false },
            itemStyle: {
                borderColor: '#fff',
                borderWidth: 1,
                gapWidth: 2
            },
            levels: [{
                itemStyle: {
                    borderColor: '#777',
                    borderWidth: 0,
                    gapWidth: 1
                },
                upperLabel: {
                    show: false
                }
            }, {
                itemStyle: {
                    borderColor: '#555',
                    borderWidth: 5,
                    gapWidth: 1
                },
                emphasis: {
                    itemStyle: {
                        borderColor: '#ddd'
                    }
                }
            }],
            data: stationData.stations.map(station => {
                const name = station.name.replace('成都', '').replace('区', '');
                return {
                    name: station.name,
                    value: station.todayEnergy || 1,
                    itemStyle: {
                        color: station.status === '正常' ? 
                            (station.usage >= 80 ? '#ff6b6b' : 
                             station.usage >= 50 ? '#4ecdc4' : '#45b7af') :
                            station.status === '维护中' ? '#ffd93d' : '#ff6b6b'
                    },
                    label: {
                        show: true,
                        position: 'inside',
                        backgroundColor: 'rgba(0, 0, 0, 0.5)',
                        borderRadius: 4,
                        padding: [4, 8],
                        formatter: function(params) {
                            const station = stationData.stations.find(s => s.name === params.name);
                            const name = station.name.replace('成都', '').replace('区', '');
                            return [
                                '{title|' + name + '}',
                                '{line1|' + station.type + '  ' + station.status + '}',
                                '{line2|' + station.ports + '  ' + station.usage + '%}'
                            ].join('\n');
                        },
                        rich: {
                            title: {
                                fontSize: 14,
                                color: '#fff',
                                fontWeight: 'bold',
                                padding: [0, 0, 4, 0],
                                align: 'center',
                                lineHeight: 20
                            },
                            line1: {
                                fontSize: 12,
                                color: '#eee',
                                padding: [4, 0],
                                align: 'center',
                                lineHeight: 16
                            },
                            line2: {
                                fontSize: 12,
                                color: '#eee',
                                padding: [4, 0],
                                align: 'center',
                                lineHeight: 16
                            }
                        }
                    }
                };
            })
        }]
    };

    monitorGrid.setOption(option);
}

// 刷新图表数据
function refreshCharts() {
    console.log('刷新图表数据...');
    
    // 模拟获取新数据
    stationData.stations.forEach(station => {
        if (station.status === '正常') {
            station.usage = Math.floor(Math.random() * 30 + 60); // 60-90%的使用率
        }
    });

    // 重新初始化所有图表
    initHeatmapChart();
    initMonitorGrid();

    // 显示提示
    showToast('数据已更新', 'success');
}

// 导出数据
function exportData() {
    console.log('导出数据...');
    
    try {
        const data = {
            timestamp: new Date().toISOString(),
            stations: stationData.stations.map(station => ({
                name: station.name,
                region: station.region,
                type: station.type,
                status: station.status,
                power: station.power,
                ports: station.ports,
                usage: station.usage
            }))
        };

        // 创建工作簿
        const wb = XLSX.utils.book_new();
        
        // 创建工作表
        const ws = XLSX.utils.json_to_sheet(data.stations);
        
        // 设置列宽
        ws['!cols'] = [
            { wch: 20 }, // 名称
            { wch: 15 }, // 区域
            { wch: 15 }, // 类型
            { wch: 15 }, // 状态
            { wch: 15 }, // 功率
            { wch: 15 }, // 端口
            { wch: 15 }  // 使用率
        ];

        // 添加工作表到工作簿
        XLSX.utils.book_append_sheet(wb, ws, "充电站数据");

        // 导出文件
        const fileName = `充电站监控数据_${new Date().toISOString().split('T')[0]}.xlsx`;
        XLSX.writeFile(wb, fileName);

        // 显示提示
        showToast('数据导出成功', 'success');
    } catch (error) {
        console.error('导出失败:', error);
        showToast('导出失败，请重试', 'danger');
    }
}

// Toast提示函数
function showToast(message, type = 'success') {
    const toast = document.getElementById('recordCancelToast');
    const toastBody = document.getElementById('recordCancelToastBody');
    
    if (toast && toastBody) {
        toastBody.textContent = message;
        toast.classList.remove('bg-success', 'bg-danger', 'bg-warning');
        toast.classList.add(`bg-${type}`);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
}

// 确保在DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM加载完成，准备初始化图表...');
    // 延迟初始化以确保页面完全加载
    setTimeout(initCharts, 500);
});

// 监听标签页切换
document.addEventListener('shown.bs.tab', function(e) {
    if (e.target.getAttribute('href') === '#charging-management') {
        console.log('切换到充电桩管理标签页，初始化图表...');
        setTimeout(initCharts, 500);
    }
});

// 导出函数到全局作用域
window.refreshCharts = refreshCharts;
window.exportData = exportData; 