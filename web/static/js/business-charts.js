// 业务处理趋势图配置
function initBusinessTrendChart() {
    var options = {
        series: [{
            name: '调度执行率',
            type: 'line',
            data: [98.5, 97.8, 99.1, 98.7, 96.5, 97.9, 98.2]
        }, {
            name: '交易成功率',
            type: 'line',
            data: [95.2, 94.8, 96.1, 95.7, 93.5, 94.9, 95.2]
        }, {
            name: '充电桩利用率',
            type: 'line',
            data: [92.5, 93.8, 91.1, 92.7, 94.5, 93.9, 92.2]
        }, {
            name: '检修完成率',
            type: 'line',
            data: [99.5, 98.8, 99.1, 99.7, 98.5, 99.9, 99.2]
        }],
        chart: {
            height: 350,
            type: 'line',
            toolbar: {
                show: true
            },
            zoom: {
                enabled: true
            }
        },
        stroke: {
            width: [3, 3, 3, 3],
            curve: 'smooth'
        },
        xaxis: {
            categories: ['3-20', '3-21', '3-22', '3-23', '3-24', '3-25', '3-26'],
            title: {
                text: '日期'
            }
        },
        yaxis: {
            title: {
                text: '完成率 (%)'
            },
            min: 90,
            max: 100
        },
        legend: {
            position: 'top',
            horizontalAlign: 'center'
        },
        markers: {
            size: 4,
            hover: {
                size: 6
            }
        },
        tooltip: {
            y: {
                formatter: function(value) {
                    return value + '%';
                }
            }
        },
        colors: ['#4F8CBE', '#2ab57d', '#fd625e', '#ffbf53']
    };

    var chart = new ApexCharts(document.querySelector("#business-trend-chart"), options);
    chart.render();
}

// 任务分布状态图配置
function initTaskDistributionChart() {
    var options = {
        series: [{
            name: '任务数量',
            data: [
                {
                    x: '电力调度',
                    y: 15,
                    goals: [
                        {
                            name: '计划数',
                            value: 18,
                            strokeColor: '#775DD0'
                        }
                    ]
                },
                {
                    x: '能源交易',
                    y: 8,
                    goals: [
                        {
                            name: '计划数',
                            value: 10,
                            strokeColor: '#775DD0'
                        }
                    ]
                },
                {
                    x: '设备检修',
                    y: 5,
                    goals: [
                        {
                            name: '计划数',
                            value: 6,
                            strokeColor: '#775DD0'
                        }
                    ]
                },
                {
                    x: '充电桩调度',
                    y: 2,
                    goals: [
                        {
                            name: '计划数',
                            value: 0,
                            strokeColor: '#775DD0'
                        }
                    ]
                }
            ]
        }],
        chart: {
            height: 350,
            type: 'bar'
        },
        plotOptions: {
            bar: {
                horizontal: false,
                columnWidth: '60%'
            }
        },
        dataLabels: {
            enabled: false
        },
        colors: ['#4F8CBE'],
        legend: {
            show: true,
            showForSingleSeries: true,
            customLegendItems: ['实际数量', '计划数量'],
            markers: {
                fillColors: ['#4F8CBE', '#775DD0']
            }
        },
        xaxis: {
            title: {
                text: '业务类型'
            }
        },
        yaxis: {
            title: {
                text: '任务数量 (个)'
            }
        },
        tooltip: {
            shared: true,
            intersect: false,
            y: {
                formatter: function(value) {
                    return value + ' 个';
                }
            }
        }
    };

    var chart = new ApexCharts(document.querySelector("#task-distribution-chart"), options);
    chart.render();
}

// 初始化所有图表
document.addEventListener('DOMContentLoaded', function() {
    initBusinessTrendChart();
    initTaskDistributionChart();
}); 