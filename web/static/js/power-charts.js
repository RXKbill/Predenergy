// 初始化时间选择器的默认值
function initDateTimeInputs() {
    const now = new Date();
    
    function setDateTimeInputs(startElement, endElement, granularity) {
        const range = getInitialDateRange(granularity);
        startElement.value = formatDateTime(range.start);
        endElement.value = formatDateTime(range.end);
    }

    // 获取当前选择的时间粒度
    const solarGranularity = document.getElementById('solarGranularity').value;
    const windGranularity = document.getElementById('windGranularity').value;

    // 设置太阳能发电的时间范围
    setDateTimeInputs(
        document.getElementById('solarStartTime'),
        document.getElementById('solarEndTime'),
        solarGranularity
    );

    // 设置风力发电的时间范围
    setDateTimeInputs(
        document.getElementById('windStartTime'),
        document.getElementById('windEndTime'),
        windGranularity
    );
}

// 格式化日期时间
function formatDateTime(date) {
    return date.toISOString().slice(0, 16);
}

// 生成模拟数据
function generateMockData(granularity, startDate, endDate, includePrediction = true, type = 'solar') {
    const data = [];
    const predictionData = [];
    let current = new Date(startDate);
    const now = new Date();

    // 对于年度和月度数据，适当扩展时间范围
    if (granularity === 'year') {
        // 确保至少有5年的数据
        const startYear = current.getFullYear();
        const endYear = new Date(endDate).getFullYear();
        if (endYear - startYear < 4) {
            current = new Date(startYear - 2, 0, 1); // 往前扩展2年
            endDate = new Date(endYear + 2, 11, 31); // 往后扩展2年
        }
    } else if (granularity === 'month') {
        // 确保至少有12个月的数据
        const monthDiff = (endDate.getFullYear() - current.getFullYear()) * 12 + 
                         (endDate.getMonth() - current.getMonth());
        if (monthDiff < 11) {
            current = new Date(current.getFullYear() - 1, current.getMonth(), 1); // 往前扩展1年
            endDate = new Date(endDate.getFullYear() + 1, endDate.getMonth(), 0); // 往后扩展1年
        }
    }

    // 根据不同的时间粒度生成数据
    while (current <= endDate) {
        let value;
        const newDate = new Date(current);
        
        if (type === 'solar') {
            // 太阳能发电数据生成逻辑
            switch(granularity) {
                case 'year':
                    const seasonalFactor = Math.sin((current.getMonth() + 1) / 12 * Math.PI) + 1;
                    value = Math.floor((Math.random() * 5000 + 8000) * seasonalFactor);
                    current.setFullYear(current.getFullYear() + 1);
                    break;
                case 'month':
                    const monthFactor = Math.sin((current.getMonth() + 1) / 12 * Math.PI) + 1;
                    value = Math.floor((Math.random() * 500 + 800) * monthFactor);
                    current.setMonth(current.getMonth() + 1);
                    break;
                case 'day':
                    // 日发电量受季节影响
                    const daySeasonFactor = Math.sin((current.getMonth() + 1) / 12 * Math.PI) + 1;
                    value = Math.floor((Math.random() * 100 + 30) * daySeasonFactor);
                    current.setDate(current.getDate() + 1);
                    break;
                case 'hour':
                    const hour = current.getHours();
                    if (hour >= 6 && hour <= 18) {
                        const hourFactor = Math.sin((hour - 6) / 12 * Math.PI);
                        value = Math.floor((Math.random() * 10 + 25) * hourFactor);
                    } else {
                        value = 0; // 夜间无发电量
                    }
                    current.setHours(current.getHours() + 1);
                    break;
            }
        } else {
            // 风力发电数据生成逻辑
            switch(granularity) {
                case 'year':
                    // 风力年发电量略低于太阳能，但波动更大
                    const yearWindFactor = 0.8 + Math.random() * 0.4; // 0.8-1.2的随机因子
                    value = Math.floor((Math.random() * 4000 + 7000) * yearWindFactor);
                    current.setFullYear(current.getFullYear() + 1);
                    break;
                case 'month':
                    // 风力月发电量波动较大
                    const monthWindFactor = 0.6 + Math.random() * 0.8; // 0.6-1.4的随机因子
                    value = Math.floor((Math.random() * 600 + 700) * monthWindFactor);
                    current.setMonth(current.getMonth() + 1);
                    break;
                case 'day':
                    // 风力日发电量波动更大
                    const windStrength = 0.4 + Math.random() * 1.2; // 0.4-1.6的随机因子
                    value = Math.floor((Math.random() * 120 + 20) * windStrength);
                    current.setDate(current.getDate() + 1);
                    break;
                case 'hour':
                    // 风力小时发电量，考虑随机波动
                    const hourWindFactor = 0.3 + Math.random() * 1.4; // 0.3-1.7的随机因子
                    value = Math.floor((Math.random() * 15 + 10) * hourWindFactor);
                    current.setHours(current.getHours() + 1);
                    break;
            }
        }

        // 修改这里：降低预测误差范围
        // 对于预测数据，根据发电类型和时间粒度调整波动范围
        let volatility;
        if (type === 'solar') {
            switch(granularity) {
                case 'hour':
                    volatility = 0.02; // 小时级预测误差 ±2%
                    break;
                case 'day':
                    volatility = 0.03; // 日级预测误差 ±3%
                    break;
                case 'month':
                    volatility = 0.04; // 月级预测误差 ±4%
                    break;
                case 'year':
                    volatility = 0.05; // 年级预测误差 ±5%
                    break;
                default:
                    volatility = 0.03;
            }
        } else {
            // 风力发电预测误差略高于太阳能
            switch(granularity) {
                case 'hour':
                    volatility = 0.03; // 小时级预测误差 ±3%
                    break;
                case 'day':
                    volatility = 0.04; // 日级预测误差 ±4%
                    break;
                case 'month':
                    volatility = 0.05; // 月级预测误差 ±5%
                    break;
                case 'year':
                    volatility = 0.06; // 年级预测误差 ±6%
                    break;
                default:
                    volatility = 0.04;
            }
        }

        // 生成更精确的预测值
        // 使用正态分布来生成预测误差，使误差更集中在中间值附近
        const randomError = (Math.random() + Math.random() + Math.random()) / 3; // 近似正态分布
        const predictionValue = value * (1 + (randomError * volatility * 2 - volatility));

        // 处理数据添加逻辑
        if (newDate > now) {
            if (includePrediction) {
                predictionData.push({
                    x: newDate.getTime(),
                    y: Math.round(predictionValue)
                });
            }
        } else {
            data.push({
                x: newDate.getTime(),
                y: value
            });
            if (includePrediction) {
                predictionData.push({
                    x: newDate.getTime(),
                    y: Math.round(predictionValue)
                });
            }
        }
    }
    
    return {
        actual: data,
        prediction: predictionData
    };
}

// 获取图表颜色配置
function getChartColorsArray(chartId) {
    const colors = document.querySelector(chartId).getAttribute('data-colors');
    if (colors) {
        return JSON.parse(colors);
    }
    return ['#4F8CBE', '#2ab57d']; // 默认颜色
}

// 将 getInitialDateRange 函数移到全局作用域
function getInitialDateRange(granularity) {
    const now = new Date();
    switch(granularity) {
        case 'year':
            return {
                start: new Date(now.getFullYear() - 2, 0, 1),
                end: new Date(now.getFullYear() + 2, 11, 31)
            };
        case 'month':
            return {
                start: new Date(now.getFullYear() - 1, now.getMonth(), 1),
                end: new Date(now.getFullYear() + 1, now.getMonth(), 0)
            };
        case 'day':
            return {
                start: new Date(now.getTime() - (30 * 24 * 60 * 60 * 1000)),
                end: new Date(now.getTime() + (15 * 24 * 60 * 60 * 1000))
            };
        case 'hour':
            return {
                start: new Date(now.getTime() - (24 * 60 * 60 * 1000)),
                end: new Date(now.getTime() + (12 * 60 * 60 * 1000))
            };
        default:
            return {
                start: new Date(now.getTime() - (30 * 24 * 60 * 60 * 1000)),
                end: new Date(now.getTime() + (15 * 24 * 60 * 60 * 1000))
            };
    }
}

// 将重复的 initCharts 函数合并为一个
function initCharts() {
    console.log('Initializing power charts...');
    
    // 检查必要的DOM元素是否存在
    const solarChartElement = document.querySelector("#solar_power_area");
    const windChartElement = document.querySelector("#wind_power_area");
    
    if (!solarChartElement || !windChartElement) {
        console.warn('Chart elements not found, skipping chart initialization');
        return;
    }
    
    // 获取图表颜色
    const solarChartColors = ['#4F8CBE', '#2ab57d']; // 默认颜色
    const windChartColors = ['#4F8CBE', '#2ab57d']; // 默认颜色
    
    try {
    // 获取当前选择的时间粒度
        const solarGranularity = document.getElementById('solarGranularity')?.value || 'day';
        const windGranularity = document.getElementById('windGranularity')?.value || 'day';

    // 获取太阳能和风力发电的初始数据
    const solarDateRange = getInitialDateRange(solarGranularity);
    const windDateRange = getInitialDateRange(windGranularity);
    
    const solarInitialData = generateMockData(solarGranularity, solarDateRange.start, solarDateRange.end);
    const windInitialData = generateMockData(windGranularity, windDateRange.start, windDateRange.end);

    const baseConfig = {
        chart: {
            height: 350,
            type: 'area',
            toolbar: {
                show: true,
                tools: {
                    download: true,
                    selection: true,
                    zoom: true,
                    zoomin: true,
                    zoomout: true,
                    pan: true,
                    reset: true
                }
                },
                background: '#2a3042'
        },
        dataLabels: { enabled: false },
        stroke: {
            curve: 'smooth',
                width: [3, 2],
                dashArray: [0, 5]
            },
            grid: { 
                borderColor: '#404040',
                xaxis: {
                    lines: {
                        show: true
                    }
                },
                yaxis: {
                    lines: {
                        show: true
                    }
                }
            },
        tooltip: {
            shared: true,
            intersect: false,
                theme: 'dark',
            y: {
                formatter: function (val) {
                    return val + " kWh";
                }
            }
        },
        xaxis: {
            type: 'datetime',
            labels: {
                    format: 'yyyy-MM-dd',
                    style: {
                        colors: '#ccc'
                    }
            }
        },
        yaxis: {
            title: {
                    text: '发电量 (kWh)',
                    style: {
                        color: '#ccc'
                    }
                },
                labels: {
                    style: {
                        colors: '#ccc'
                    }
            }
        },
        legend: {
            show: true,
                position: 'top',
                labels: {
                    colors: '#ccc'
                }
        }
    };

        // 初始化太阳能发电图表
        window.solarChart = new ApexCharts(
            solarChartElement,
            {
                ...baseConfig,
                colors: solarChartColors,
                series: [
                    {
                        name: '实际发电量',
                        data: solarInitialData.actual
                    },
                    {
                        name: '预测发电量',
                        data: solarInitialData.prediction
                    }
                ]
            }
        );

        // 初始化风力发电图表
        window.windChart = new ApexCharts(
            windChartElement,
            {
                ...baseConfig,
                colors: windChartColors,
                series: [
                    {
                        name: '实际发电量',
                        data: windInitialData.actual
                    },
                    {
                        name: '预测发电量',
                        data: windInitialData.prediction
                    }
                ]
            }
        );

        // 渲染图表
        solarChart.render();
        windChart.render();

        // 更新图表配置以匹配时间粒度
        updateChart('solar_power_area', solarInitialData, solarGranularity);
        updateChart('wind_power_area', windInitialData, windGranularity);

        console.log('Charts initialized successfully');
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
}

// 确保在DOM加载完成后初始化图表
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing charts...');
    
    // 延迟一点初始化，确保所有元素都已经准备好
    setTimeout(() => {
        try {
            initDateTimeInputs();
            initCharts();
            
            // 添加查询按钮事件监听器
            const solarQueryBtn = document.getElementById('solarQuery');
            if (solarQueryBtn) {
                solarQueryBtn.addEventListener('click', function() {
                    const granularity = document.getElementById('solarGranularity')?.value || 'day';
                    const startTime = document.getElementById('solarStartTime')?.value;
                    const endTime = document.getElementById('solarEndTime')?.value;
                    if (startTime && endTime) {
                        updateChartData('solar', granularity, startTime, endTime);
                    }
                });
            }

            const windQueryBtn = document.getElementById('windQuery');
            if (windQueryBtn) {
                windQueryBtn.addEventListener('click', function() {
                    const granularity = document.getElementById('windGranularity')?.value || 'day';
                    const startTime = document.getElementById('windStartTime')?.value;
                    const endTime = document.getElementById('windEndTime')?.value;
                    if (startTime && endTime) {
                        updateChartData('wind', granularity, startTime, endTime);
                    }
                });
            }

            // 修改时间粒度变化事件监听器
            const solarGranularity = document.getElementById('solarGranularity');
            if (solarGranularity) {
                solarGranularity.addEventListener('change', function() {
                    handleGranularityChange('solar');
                });
            }

            const windGranularity = document.getElementById('windGranularity');
            if (windGranularity) {
                windGranularity.addEventListener('change', function() {
                    handleGranularityChange('wind');
                });
            }

            // 添加重置按钮事件监听器
            const resetZoomBtn = document.getElementById('resetZoom');
            if (resetZoomBtn) {
                resetZoomBtn.addEventListener('click', function() {
                    if(window.solarChart) {
                        window.solarChart.resetSeries();
                    }
                });
            }

            const resetWindZoomBtn = document.getElementById('resetWindZoom');
            if (resetWindZoomBtn) {
                resetWindZoomBtn.addEventListener('click', function() {
                    if(window.windChart) {
                        window.windChart.resetSeries();
                    }
                });
            }

            // 为表格按钮添加点击事件监听器
            const periodButtons = document.querySelectorAll('.table-period-btn');
            periodButtons.forEach(button => {
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    // 移除所有按钮的 active 类
                    periodButtons.forEach(btn => btn.classList.remove('active'));
                    // 为当前点击的按钮添加 active 类
                    this.classList.add('active');
                    
                    const period = this.getAttribute('data-period');
                    console.log('Updating table for period:', period);
                    updateTablePeriod(period);
                });
            });

            // 初始化表格显示（默认显示日数据）
            const defaultPeriod = 'day';
            const defaultButton = document.querySelector(`.table-period-btn[data-period="${defaultPeriod}"]`);
            if (defaultButton) {
                defaultButton.classList.add('active');
                updateTablePeriod(defaultPeriod);
            }

            console.log('All chart components initialized successfully');
        } catch (error) {
            console.error('Error in chart initialization:', error);
        }
    }, 500); // 延迟500ms初始化
});

// 修改 updateChartData 函数
function updateChartData(chartType, granularity, startTime, endTime) {
    try {
        const start = new Date(startTime);
        const end = new Date(endTime);
        
        if (start > end) {
            alert('开始时间不能大于结束时间');
            return;
        }
        
        // 确保结束时间包含足够的预测时间
        const now = new Date();
        let adjustedEnd = end;
        
        // 根据粒度调整结束时间以包含预测数据
        switch(granularity) {
            case 'year':
                adjustedEnd = new Date(end.getFullYear() + 2, 11, 31);
                break;
            case 'month':
                adjustedEnd = new Date(end.getFullYear() + 1, end.getMonth(), 0);
                break;
            case 'day':
                adjustedEnd = new Date(end.getTime() + (15 * 24 * 60 * 60 * 1000));
                break;
            case 'hour':
                adjustedEnd = new Date(end.getTime() + (12 * 60 * 60 * 1000));
                break;
        }
        
        const data = generateMockData(granularity, start, adjustedEnd, true, chartType);
        const chartId = chartType === 'solar' ? 'solar_power_area' : 'wind_power_area';
        updateChart(chartId, data, granularity);
    } catch (error) {
        console.error('Error updating chart data:', error);
        alert('更新数据失败，请检查时间范围是否正确');
    }
}

// 修改 updateTablePeriod 函数
function updateTablePeriod(period) {
    console.log('Starting table update for period:', period);
    
    // 获取当前活动的图表类型（太阳能或风力）
    const type = document.querySelector('#solar_power_area').style.display !== 'none' ? 'solar' : 'wind';
    console.log('Current chart type:', type);
    
    // 获取日期范围
    const range = getInitialDateRange(period);
    console.log('Date range:', range);
    
    // 生成新数据
    const data = generateMockData(period, range.start, range.end, true, type);
    console.log('Generated data:', data);
    
    // 更新表格
    updatePredictionTable(data, type, period);
}

// 添加表格更新相关函数
function updatePredictionTable(data, type, granularity = 'day') {
    console.log('Updating prediction table:', {type, granularity});
    
    const tbody = document.querySelector('#predictionDetailsTable tbody');
    if (!tbody) {
        console.error('Prediction table tbody not found');
        return;
    }

    tbody.innerHTML = '';

    // 获取当前时间
    const now = new Date();

    // 根据时间粒度选择显示的记录数和时间范围
    let timeRange;
    switch(granularity) {
        case 'year':
            timeRange = {
                start: new Date(now.getFullYear() - 2, 0, 1),
                end: new Date(now.getFullYear() + 2, 11, 31)
            };
            break;
        case 'month':
            timeRange = {
                start: new Date(now.getFullYear(), now.getMonth() - 6, 1),
                end: new Date(now.getFullYear(), now.getMonth() + 6, 0)
            };
            break;
        case 'day':
            timeRange = {
                start: new Date(now.getTime() - (7 * 24 * 60 * 60 * 1000)),
                end: new Date(now.getTime() + (7 * 24 * 60 * 60 * 1000))
            };
            break;
        case 'hour':
            timeRange = {
                start: new Date(now.getTime() - (12 * 60 * 60 * 1000)),
                end: new Date(now.getTime() + (12 * 60 * 60 * 1000))
            };
            break;
    }

    // 过滤并组织数据
    const tableData = [];
    const actualMap = new Map(data.actual.map(item => [new Date(item.x).getTime(), item]));
    const predictionMap = new Map(data.prediction.map(item => [new Date(item.x).getTime(), item]));

    // 遍历时间范围内的每个时间点
    let current = new Date(timeRange.start);
    while (current <= timeRange.end) {
        const timestamp = current.getTime();
        const actual = actualMap.get(timestamp);
        const prediction = predictionMap.get(timestamp);

        if (prediction) {
            const actualValue = actual ? actual.y : null;
            const predictedValue = prediction.y;
            
            // 只计算已有实际值的误差率
            let errorRate = null;
            if (actualValue !== null && actualValue !== 0) {
                errorRate = Math.abs((predictedValue - actualValue) / actualValue * 100);
            }

            // 确定状态
            let status = '';
            let statusClass = '';
            if (errorRate === null) {
                status = '预测中';
                statusClass = 'status-warning';
            } else if (errorRate <= 3) {
                status = '正常';
                statusClass = 'status-normal';
            } else if (errorRate <= 6) {
                status = '偏差';
                statusClass = 'status-warning';
            } else {
                status = '异常';
                statusClass = 'status-danger';
            }

            // 格式化时间显示
            let formattedDate;
            switch(granularity) {
                case 'year':
                    formattedDate = current.getFullYear() + '年';
                    break;
                case 'month':
                    formattedDate = `${current.getFullYear()}年${current.getMonth() + 1}月`;
                    break;
                case 'day':
                    formattedDate = `${current.getMonth() + 1}月${current.getDate()}日`;
                    break;
                case 'hour':
                    formattedDate = `${current.getHours().toString().padStart(2, '0')}:00`;
                    break;
            }

            tableData.push({
                time: formattedDate,
                predicted: predictedValue.toFixed(1),
                actual: actualValue !== null ? actualValue.toFixed(1) : '待更新',
                errorRate: errorRate !== null ? errorRate.toFixed(2) : '-',
                status: status,
                statusClass: statusClass
            });
        }

        // 根据粒度更新时间
        switch(granularity) {
            case 'year':
                current.setFullYear(current.getFullYear() + 1);
                break;
            case 'month':
                current.setMonth(current.getMonth() + 1);
                break;
            case 'day':
                current.setDate(current.getDate() + 1);
                break;
            case 'hour':
                current.setHours(current.getHours() + 1);
                break;
        }
    }

    // 生成表格行
    tableData.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.time}</td>
            <td>${row.predicted} kW</td>
            <td>${row.actual === '待更新' ? row.actual : row.actual + ' kW'}</td>
            <td>${row.errorRate}%</td>
            <td><span class="status-indicator ${row.statusClass}"></span>${row.status}</td>
        `;
        tbody.appendChild(tr);
    });

    // 更新表格标题
    const tableTitle = document.querySelector('#predictionTableTitle');
    if (tableTitle) {
        const periodText = {
            year: '年度',
            month: '月度',
            day: '日',
            hour: '小时'
        }[granularity] || '';
        tableTitle.textContent = `${type === 'solar' ? '太阳能' : '风力'}发电${periodText}预测详情`;
    }
}

// 修改 updateChart 函数，添加表格更新
function updateChart(chartId, data, granularity) {
    let xaxisFormat;
    let tooltipFormat;
    
    switch(granularity) {
        case 'year':
            xaxisFormat = 'yyyy';
            tooltipFormat = 'yyyy';
            break;
        case 'month':
            xaxisFormat = 'yyyy-MM';
            tooltipFormat = 'yyyy-MM';
            break;
        case 'day':
            xaxisFormat = 'MM-dd';
            tooltipFormat = 'yyyy-MM-dd';
            break;
        case 'hour':
            xaxisFormat = 'MM-dd HH:mm';
            tooltipFormat = 'yyyy-MM-dd HH:mm';
            break;
    }

    const chart = chartId === 'solar_power_area' ? window.solarChart : window.windChart;
    
    if (chart) {
        chart.updateOptions({
            xaxis: {
                type: 'datetime',
                labels: {
                    format: xaxisFormat,
                    datetimeUTC: false,
                    rotate: granularity === 'hour' ? -45 : 0,
                    style: {
                        fontSize: '12px'
                    }
                },
                tickAmount: granularity === 'hour' ? 24 : undefined
            },
            tooltip: {
                x: {
                    format: tooltipFormat
                }
            }
        });
        
        chart.updateSeries([
            {
                name: '实际发电量',
                data: data.actual
            },
            {
                name: '预测发电量',
                data: data.prediction
            }
        ]);

        // 更新预测详情表格
        updatePredictionTable(data, chartId === 'solar_power_area' ? 'solar' : 'wind', granularity);
    }
}

// 修改时间粒度变化的事件处理
function handleGranularityChange(type) {
    const granularity = document.getElementById(type + 'Granularity').value;
    const range = getInitialDateRange(granularity);
    
    // 更新时间输入框
    document.getElementById(type + 'StartTime').value = formatDateTime(range.start);
    document.getElementById(type + 'EndTime').value = formatDateTime(range.end);
    
    // 生成新的数据
    const data = generateMockData(granularity, range.start, range.end, true, type);
    
    // 更新图表
    const chartId = type + '_power_area';
    updateChart(chartId, data, granularity);
    
    // 更新表格数据
    updatePredictionTable(data, type, granularity);
    
    // 同步更新表格按钮状态
    const periodButtons = document.querySelectorAll('.table-period-btn');
    periodButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-period') === granularity) {
            btn.classList.add('active');
        }
    });
}

// 事件监听器设置
document.addEventListener('DOMContentLoaded', function() {
    try {
        console.log('DOM Content Loaded');
        initDateTimeInputs();
        initCharts();

        // 添加查询按钮事件监听器
        document.getElementById('solarQuery').addEventListener('click', function() {
            const granularity = document.getElementById('solarGranularity').value;
            const startTime = document.getElementById('solarStartTime').value;
            const endTime = document.getElementById('solarEndTime').value;
            updateChartData('solar', granularity, startTime, endTime);
        });

        document.getElementById('windQuery').addEventListener('click', function() {
            const granularity = document.getElementById('windGranularity').value;
            const startTime = document.getElementById('windStartTime').value;
            const endTime = document.getElementById('windEndTime').value;
            updateChartData('wind', granularity, startTime, endTime);
        });

        // 修改时间粒度变化事件监听器
        document.getElementById('solarGranularity').addEventListener('change', function() {
            handleGranularityChange('solar');
        });

        document.getElementById('windGranularity').addEventListener('change', function() {
            handleGranularityChange('wind');
        });

        // 添加重置按钮事件监听器
        document.getElementById('resetZoom').addEventListener('click', function() {
            if(window.solarChart) {
                window.solarChart.resetSeries();
            }
        });

        document.getElementById('resetWindZoom').addEventListener('click', function() {
            if(window.windChart) {
                window.windChart.resetSeries();
            }
        });

        // 为表格按钮添加点击事件监听器
        const periodButtons = document.querySelectorAll('.table-period-btn');
        periodButtons.forEach(button => {
            button.addEventListener('click', function(e) {
                e.preventDefault();
                
                // 移除所有按钮的 active 类
                periodButtons.forEach(btn => btn.classList.remove('active'));
                // 为当前点击的按钮添加 active 类
                this.classList.add('active');
                
                const period = this.getAttribute('data-period');
                console.log('Updating table for period:', period); // 调试日志
                updateTablePeriod(period);
            });
        });

        // 初始化表格显示（默认显示日数据）
        const defaultPeriod = 'day';
        const defaultButton = document.querySelector(`.table-period-btn[data-period="${defaultPeriod}"]`);
        if (defaultButton) {
            defaultButton.classList.add('active');
            updateTablePeriod(defaultPeriod);
        }

    } catch (error) {
        console.error('Error in DOMContentLoaded:', error);
    }
}); 