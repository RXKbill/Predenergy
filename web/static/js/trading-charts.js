/**
 * 能源交易图表配置和初始化
 */

class TradingCharts {
    constructor() {
        // 添加标志位，防止重复初始化
        if (window.tradingChartsInstance) {
            return window.tradingChartsInstance;
        }
        window.tradingChartsInstance = this;

        // 添加分页配置
        this.pagination = {
            currentPage: 1,
            pageSize: 6,
            totalPages: 0
        };

        this.chartData = this.generateMockData();
        
        // 确保在DOM加载完成后再初始化图表
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.initializeCharts();
                this.initializeSparklines();
                this.initializeStatusIcon();
                this.initializeStrategyHandlers();
            });
        } else {
            this.initializeCharts();
            this.initializeSparklines();
            this.initializeStatusIcon();
            this.initializeStrategyHandlers();
        }
        
        this.transactions = this._generateMockTransactions();
        this.initializeTransactionHandlers();
    }

    /**
     * 生成模拟数据
     */
    generateMockData() {
        const data = [];
        let date = new Date();
        date.setHours(0, 0, 0, 0);
        
        for (let i = 0; i < 100; i++) {
            const basePrice = 480;
            const volatility = 20;
            const open = basePrice + (Math.random() - 0.5) * volatility;
            const close = basePrice + (Math.random() - 0.5) * volatility;
            const high = Math.max(open, close) + Math.random() * volatility/2;
            const low = Math.min(open, close) - Math.random() * volatility/2;
            const volume = Math.floor(Math.random() * 1000) + 500;

            data.push({
                x: new Date(date.getTime() + i * 3600000), // 每小时一个数据点
                y: [open, high, low, close],
                volume: volume
            });
        }
        return data;
    }

    /**
     * 初始化图表
     */
    initializeCharts() {
        // 检查元素是否存在
        const tradingChartElement = document.querySelector("#tradingChart");
        const volumeChartElement = document.querySelector("#volumeChart");
        
        if (!tradingChartElement || !volumeChartElement) {
            console.warn('Trading chart elements not found, skipping chart initialization');
            return;
        }

        // K线图配置
        const candlestickOptions = {
            series: [{
                name: '电力价格',
                data: this.chartData.map(item => ({
                    x: item.x,
                    y: item.y
                }))
            }],
            chart: {
                type: 'candlestick',
                height: 400,
                toolbar: {
                    show: true,
                    tools: {
                        download: false,
                        selection: true,
                        zoom: true,
                        zoomin: true,
                        zoomout: true,
                        pan: true,
                    }
                },
                background: '#2a3042',
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 800
                }
            },
            plotOptions: {
                candlestick: {
                    colors: {
                        upward: '#26a69a',
                        downward: '#ef5350'
                    },
                    wick: {
                        useFillColor: true
                    }
                }
            },
            title: {
                text: '电力现货价格走势',
                align: 'left',
                style: {
                    fontSize: '16px',
                    color: '#cccccc'
                }
            },
            xaxis: {
                type: 'datetime',
                labels: {
                    style: {
                        colors: '#ccc'
                    },
                    datetimeFormatter: {
                        year: 'yyyy年',
                        month: 'MM月',
                        day: 'dd日',
                        hour: 'HH:mm'
                    }
                },
                title: {
                    text: '交易时间',
                    style: {
                        color: '#cccccc'
                    }
                }
            },
            yaxis: {
                tooltip: {
                    enabled: true
                },
                labels: {
                    style: {
                        colors: '#ccc'
                    },
                    formatter: function (value) {
                        return '¥' + value.toFixed(2);
                    }
                },
                title: {
                    text: '电力价格 (元/MWh)',
                    style: {
                        color: '#cccccc'
                    }
                }
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
                theme: 'dark',
                custom: function({ seriesIndex, dataPointIndex, w }) {
                    const o = w.globals.seriesCandleO[seriesIndex][dataPointIndex];
                    const h = w.globals.seriesCandleH[seriesIndex][dataPointIndex];
                    const l = w.globals.seriesCandleL[seriesIndex][dataPointIndex];
                    const c = w.globals.seriesCandleC[seriesIndex][dataPointIndex];
                    const date = new Date(w.globals.seriesX[seriesIndex][dataPointIndex]);
                    
                    return `
                    <div class="p-2">
                        <div class="mb-2">${date.toLocaleString('zh-CN')}</div>
                        <div class="d-flex justify-content-between mb-1">
                            <span>开盘价：</span>
                            <span class="ms-2">¥${o.toFixed(2)}</span>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span>最高价：</span>
                            <span class="ms-2">¥${h.toFixed(2)}</span>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span>最低价：</span>
                            <span class="ms-2">¥${l.toFixed(2)}</span>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span>收盘价：</span>
                            <span class="ms-2">¥${c.toFixed(2)}</span>
                        </div>
                    </div>`;
                }
            }
        };

        // 成交量图配置
        const volumeOptions = {
            series: [{
                name: '成交电量',
                data: this.chartData.map(item => ({
                    x: item.x,
                    y: item.volume
                }))
            }],
            chart: {
                height: 100,
                type: 'bar',
                brush: {
                    enabled: true,
                    target: 'tradingChart'
                },
                background: '#2a3042'
            },
            plotOptions: {
                bar: {
                    colors: {
                        ranges: [{
                            from: -1000000,
                            to: 0,
                            color: '#ef5350'
                        }, {
                            from: 0,
                            to: 1000000,
                            color: '#26a69a'
                        }]
                    }
                }
            },
            dataLabels: {
                enabled: false
            },
            stroke: {
                width: 0
            },
            xaxis: {
                type: 'datetime',
                labels: {
                    style: {
                        colors: '#ccc'
                    },
                    datetimeFormatter: {
                        year: 'yyyy年',
                        month: 'MM月',
                        day: 'dd日',
                        hour: 'HH:mm'
                    }
                }
            },
            yaxis: {
                labels: {
                    style: {
                        colors: '#ccc'
                    },
                    formatter: function (value) {
                        return value.toFixed(0) + ' MWh';
                    }
                },
                title: {
                    text: '成交电量 (MWh)',
                    style: {
                        color: '#cccccc'
                    }
                }
            },
            grid: {
                borderColor: '#404040'
            },
            tooltip: {
                theme: 'dark',
                y: {
                    formatter: function(value) {
                        return value.toFixed(0) + ' MWh';
                    }
                }
            }
        };

        try {
            // 创建图表实例
            this.tradingChart = new ApexCharts(tradingChartElement, candlestickOptions);
            this.volumeChart = new ApexCharts(volumeChartElement, volumeOptions);

            // 渲染图表
            this.tradingChart.render();
            this.volumeChart.render();
            
            console.log('Charts initialized successfully');
        } catch (error) {
            console.error('Error initializing charts:', error);
        }
    }

    /**
     * 更新图表周期
     */
    updatePeriod(period) {
        let filteredData = [...this.chartData];
        let periodText = '';
        
        switch(period) {
            case '1D':
                filteredData = this.chartData.slice(-24);
                periodText = '24小时';
                break;
            case '7D':
                filteredData = this.chartData.slice(-7 * 24);
                periodText = '7天';
                break;
            case '1M':
                filteredData = this.chartData.slice(-30 * 24);
                periodText = '30天';
                break;
            case 'ALL':
                periodText = '全部';
                break;
        }
        
        // 更新K线图
        this.tradingChart.updateOptions({
            title: {
                text: `电力现货价格走势 (${periodText})`
            }
        });
        
        this.tradingChart.updateSeries([{
            data: filteredData.map(item => ({
                x: item.x,
                y: item.y
            }))
        }]);
        
        // 更新成交量图
        this.volumeChart.updateSeries([{
            data: filteredData.map(item => ({
                x: item.x,
                y: item.volume
            }))
        }]);
    }

    /**
     * 初始化迷你图表
     */
    initializeSparklines() {
        const volumeSparkElement = document.querySelector("#trading-volume-spark");
        const profitSparkElement = document.querySelector("#profit-forecast-spark");
        
        if (!volumeSparkElement || !profitSparkElement) {
            console.warn('Sparkline elements not found, skipping sparkline initialization');
            return;
        }

        try {
            // 交易量迷你图表配置
            const volumeSparkOptions = {
                series: [{
                    name: '交易量',
                    data: this.generateSparklineData()
                }],
                chart: {
                    type: 'area',
                    height: 40,
                    sparkline: {
                        enabled: true
                    },
                    animations: {
                        enabled: true,
                        easing: 'linear',
                        speed: 300
                    }
                },
                stroke: {
                    curve: 'smooth',
                    width: 2
                },
                fill: {
                    opacity: 0.3,
                    gradient: {
                        enabled: true,
                        shadeIntensity: 0.5,
                        inverseColors: false,
                        opacityFrom: 0.5,
                        opacityTo: 0.1,
                    }
                },
                colors: ['#34c38f'],
                tooltip: {
                    fixed: {
                        enabled: false
                    },
                    x: {
                        show: false
                    },
                    y: {
                        title: {
                            formatter: function (seriesName) {
                                return '交易量：'
                            }
                        },
                        formatter: (value) => `${value} MWh`
                    },
                    marker: {
                        show: false
                    }
                }
            };

            // 收益预测迷你图表配置
            const profitSparkOptions = {
                series: [{
                    name: '预计收益',
                    data: this.generateProfitData()
                }],
                chart: {
                    type: 'bar',
                    height: 40,
                    sparkline: {
                        enabled: true
                    },
                    animations: {
                        enabled: true,
                        easing: 'linear',
                        speed: 300
                    }
                },
                plotOptions: {
                    bar: {
                        columnWidth: '60%',
                        colors: {
                            ranges: [{
                                from: -100000,
                                to: 0,
                                color: '#ef5350'
                            }, {
                                from: 0,
                                to: 100000,
                                color: '#556ee6'
                            }]
                        }
                    }
                },
                colors: ['#556ee6'],
                tooltip: {
                    fixed: {
                        enabled: false
                    },
                    x: {
                        show: false
                    },
                    y: {
                        title: {
                            formatter: function (seriesName) {
                                return '预计收益：'
                            }
                        },
                        formatter: (value) => `¥${value.toLocaleString()}`
                    },
                    marker: {
                        show: false
                    }
                }
            };

            // 初始化交易量迷你图表
            this.volumeSparkChart = new ApexCharts(volumeSparkElement, volumeSparkOptions);
            this.volumeSparkChart.render();

            // 初始化收益预测迷你图表
            this.profitSparkChart = new ApexCharts(profitSparkElement, profitSparkOptions);
            this.profitSparkChart.render();

            // 启动定时更新
            this.startSparklineUpdates();
        } catch (error) {
            console.error('Error initializing sparklines:', error);
        }
    }

    /**
     * 生成迷你图表数据
     */
    generateSparklineData() {
        const data = [];
        for (let i = 0; i < 10; i++) {
            data.push(Math.floor(Math.random() * 500) + 1000); // 1000-1500范围内的随机数
        }
        return data;
    }

    /**
     * 生成收益预测数据
     */
    generateProfitData() {
        const data = [];
        for (let i = 0; i < 10; i++) {
            data.push(Math.floor(Math.random() * 50000) + 100000); // 100000-150000范围内的随机数
        }
        return data;
    }

    /**
     * 启动迷你图表实时更新
     */
    startSparklineUpdates() {
        setInterval(() => {
            // 更新交易量数据
            const volumeData = this.volumeSparkChart.w.globals.series[0];
            volumeData.shift();
            volumeData.push(Math.floor(Math.random() * 500) + 1000);
            
            this.volumeSparkChart.updateSeries([{
                data: volumeData
            }]);

            // 更新收益预测数据
            const profitData = this.profitSparkChart.w.globals.series[0];
            profitData.shift();
            profitData.push(Math.floor(Math.random() * 50000) + 100000);
            
            this.profitSparkChart.updateSeries([{
                data: profitData
            }]);

            // 更新执行率状态
            this.updateExecutionRate();
        }, 3000); // 每3秒更新一次
    }

    /**
     * 初始化状态图标
     */
    initializeStatusIcon() {
        const statusIcon = document.querySelector('.status-icon.success');
        if (statusIcon) {
            statusIcon.innerHTML = '<i class="ri-check-line"></i>';
            this.updateExecutionRate();
        }
    }

    /**
     * 更新执行率状态
     */
    updateExecutionRate() {
        const executionRate = document.querySelector('.status-icon.success');
        if (executionRate) {
            const rate = (Math.random() * 5 + 95).toFixed(1); // 95-100之间的随机数
            const parentElement = executionRate.closest('.card-body');
            if (parentElement) {
                const rateElement = parentElement.querySelector('h3');
                if (rateElement) {
                    rateElement.textContent = `${rate}%`;
                }
            }
        }
    }

    /**
     * 初始化策略相关的事件处理
     */
    initializeStrategyHandlers() {
        // 移除可能存在的旧事件监听器
        const saveConfigBtn = document.querySelector('#strategyConfigModal .btn-primary');
        const createStrategyBtn = document.querySelector('#newStrategyModal .btn-primary');

        if (saveConfigBtn) {
            // 移除旧的事件监听器
            const newSaveConfigBtn = saveConfigBtn.cloneNode(true);
            saveConfigBtn.parentNode.replaceChild(newSaveConfigBtn, saveConfigBtn);
            // 添加新的事件监听器
            newSaveConfigBtn.addEventListener('click', () => this.saveStrategyConfig());
        }

        if (createStrategyBtn) {
            // 移除旧的事件监听器
            const newCreateStrategyBtn = createStrategyBtn.cloneNode(true);
            createStrategyBtn.parentNode.replaceChild(newCreateStrategyBtn, createStrategyBtn);
            // 添加新的事件监听器
            newCreateStrategyBtn.addEventListener('click', () => this.createNewStrategy());
        }
    }

    /**
     * 保存策略配置
     */
    saveStrategyConfig() {
        // 获取策略配置表单数据
        const peakStartTime = document.querySelector('#strategyConfigModal input[type="time"]:first-of-type').value;
        const peakEndTime = document.querySelector('#strategyConfigModal input[type="time"]:last-of-type').value;
        const minPriceDiff = document.querySelector('#strategyConfigModal input[type="number"]:nth-of-type(1)').value;
        const maxVolume = document.querySelector('#strategyConfigModal input[type="number"]:nth-of-type(2)').value;
        const takeProfit = document.querySelector('#strategyConfigModal input[type="number"]:nth-of-type(3)').value;
        const stopLoss = document.querySelector('#strategyConfigModal input[type="number"]:nth-of-type(4)').value;

        // 验证输入
        if (!peakStartTime || !peakEndTime || !minPriceDiff || !maxVolume || !takeProfit || !stopLoss) {
            this._showToast('请填写所有必要的配置项', 'warning');
            return;
        }

        // 构建配置对象
        const config = {
            peakTime: {
                start: peakStartTime,
                end: peakEndTime
            },
            minPriceDiff: parseFloat(minPriceDiff),
            maxVolume: parseFloat(maxVolume),
            takeProfit: parseFloat(takeProfit),
            stopLoss: parseFloat(stopLoss)
        };

        // 这里可以添加保存到后端的逻辑
        console.log('保存策略配置:', config);

        // 显示成功提示
        this._showToast('策略配置已保存', 'success');

        // 关闭模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('strategyConfigModal'));
        modal.hide();
    }

    /**
     * 创建新策略
     */
    createNewStrategy() {
        // 获取表单数据
        const name = document.querySelector('#newStrategyModal input[type="text"]').value;
        const type = document.querySelector('#newStrategyModal select:first-of-type').value;
        const description = document.querySelector('#newStrategyModal textarea').value;
        const initialFund = document.querySelector('#newStrategyModal input[type="number"]').value;
        const riskLevel = document.querySelector('#newStrategyModal select:last-of-type').value;

        // 验证输入
        if (!name || !type || !description || !initialFund || !riskLevel) {
            this._showToast('请填写所有必要的策略信息', 'warning');
            return;
        }

        // 构建策略对象
        const strategy = {
            id: 'STR' + Date.now(), // 添加唯一ID
            name,
            type,
            description,
            initialFund: parseFloat(initialFund),
            riskLevel,
            createTime: new Date().toISOString()
        };

        // 这里可以添加保存到后端的逻辑
        console.log('创建新策略:', strategy);

        // 显示成功提示
        this._showToast('策略创建成功', 'success');

        // 清空表单
        this._resetNewStrategyForm();

        // 关闭模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('newStrategyModal'));
        modal.hide();

        // 刷新策略列表
        this._refreshStrategyList(strategy);
    }

    /**
     * 重置新建策略表单
     */
    _resetNewStrategyForm() {
        const form = document.querySelector('#newStrategyModal form');
        if (form) {
            form.reset();
        } else {
            // 如果没有form标签，手动重置各个输入框
            document.querySelector('#newStrategyModal input[type="text"]').value = '';
            document.querySelector('#newStrategyModal select:first-of-type').selectedIndex = 0;
            document.querySelector('#newStrategyModal textarea').value = '';
            document.querySelector('#newStrategyModal input[type="number"]').value = '';
            document.querySelector('#newStrategyModal select:last-of-type').selectedIndex = 0;
        }
    }

    /**
     * 刷新策略列表
     */
    _refreshStrategyList(newStrategy) {
        const strategyList = document.querySelector('.strategy-list');
        if (!strategyList) return;

        // 检查是否已存在相同ID的策略
        const existingStrategy = document.querySelector(`[data-strategy-id="${newStrategy.id}"]`);
        if (existingStrategy) {
            return; // 如果已存在，不重复添加
        }
        
        // 创建新的策略项
        const strategyItem = document.createElement('div');
        strategyItem.className = 'strategy-item';
        strategyItem.setAttribute('data-strategy-id', newStrategy.id); // 添加策略ID属性
        strategyItem.innerHTML = `
            <div class="strategy-info">
                <h6>${newStrategy.name}</h6>
                <p>${newStrategy.description}</p>
                <div class="strategy-meta">
                    <div class="strategy-meta-item">
                        <i class="ri-line-chart-line"></i>
                        <span>收益率: 0.0%</span>
                    </div>
                    <div class="strategy-meta-item">
                        <i class="ri-time-line"></i>
                        <span>运行时间: 0天</span>
                    </div>
                    <div class="strategy-meta-item">
                        <i class="ri-exchange-funds-line"></i>
                        <span>成交次数: 0次</span>
                    </div>
                </div>
            </div>
            <div class="strategy-actions">
                <div class="strategy-status testing">测试中</div>
                <button type="button" class="strategy-config-btn" data-bs-toggle="modal" data-bs-target="#strategyConfigModal">
                    <i class="ri-settings-3-line"></i>
                </button>
            </div>
        `;

        // 添加到列表开头
        strategyList.insertBefore(strategyItem, strategyList.firstChild);
    }

    /**
     * 显示提示信息
     */
    _showToast(message, type = 'success') {
        const toastContainer = document.createElement('div');
        toastContainer.className = `toast bg-${type} text-white`;
        toastContainer.setAttribute('role', 'alert');
        toastContainer.setAttribute('aria-live', 'assertive');
        toastContainer.setAttribute('aria-atomic', 'true');
        
        toastContainer.innerHTML = `
            <div class="toast-header bg-${type} text-white">
                <strong class="me-auto">系统提示</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        `;

        document.body.appendChild(toastContainer);
        const toast = new bootstrap.Toast(toastContainer);
        toast.show();

        // 自动移除toast元素
        toastContainer.addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toastContainer);
        });
    }

    /**
     * 生成模拟交易记录
     */
    _generateMockTransactions() {
        const types = ['现货交易', '合约交易', '辅助服务'];
        const statuses = [
            { status: '已成交', class: 'success' },
            { status: '待确认', class: 'warning' },
            { status: '已撤销', class: 'danger' },
            { status: '执行中', class: 'info' }
        ];
        
        const transactions = [];
        const startDate = new Date('2024-03-01');
        const endDate = new Date();

        for (let i = 0; i < 20; i++) {
            const type = types[Math.floor(Math.random() * types.length)];
            const isBuy = Math.random() > 0.5;
            const price = 450 + Math.random() * 100;
            const volume = Math.floor(Math.random() * 500) + 50;
            const amount = price * volume;
            const status = statuses[Math.floor(Math.random() * statuses.length)];
            const date = new Date(startDate.getTime() + Math.random() * (endDate.getTime() - startDate.getTime()));

            transactions.push({
                id: `TRD-${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, '0')}${String(date.getDate()).padStart(2, '0')}${String(i + 1).padStart(3, '0')}`,
                type: type,
                direction: isBuy ? '买入' : '卖出',
                price: price.toFixed(2),
                volume: volume,
                amount: amount.toFixed(2),
                status: status.status,
                statusClass: status.class,
                timestamp: date.toISOString(),
                strategy: Math.random() > 0.7 ? '峰谷套利策略' : (Math.random() > 0.5 ? '价差套利策略' : '调频辅助服务策略')
            });
        }

        // 按时间倒序排序
        return transactions.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    }

    /**
     * 初始化交易记录相关的事件处理
     */
    initializeTransactionHandlers() {
        // 绑定筛选按钮事件
        const filterBtn = document.querySelector('.table-trading').closest('.card').querySelector('.ri-filter-3-line').closest('button');
        if (filterBtn) {
            filterBtn.addEventListener('click', () => this._showTransactionFilter());
        }

        // 绑定导出按钮事件
        const exportBtn = document.querySelector('.table-trading').closest('.card').querySelector('.ri-download-2-line').closest('button');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this._exportTransactions());
        }

        // 初始显示交易记录
        this._updateTransactionTable();
    }

    /**
     * 更新交易记录表格
     */
    _updateTransactionTable(filters = {}) {
        const tbody = document.querySelector('.table-trading tbody');
        if (!tbody) return;

        // 应用筛选
        let filteredTransactions = this.transactions;
        if (filters.type) {
            filteredTransactions = filteredTransactions.filter(t => t.type === filters.type);
        }
        if (filters.direction) {
            filteredTransactions = filteredTransactions.filter(t => t.direction === filters.direction);
        }
        if (filters.status) {
            filteredTransactions = filteredTransactions.filter(t => t.status === filters.status);
        }
        if (filters.startDate) {
            filteredTransactions = filteredTransactions.filter(t => new Date(t.timestamp) >= new Date(filters.startDate));
        }
        if (filters.endDate) {
            filteredTransactions = filteredTransactions.filter(t => new Date(t.timestamp) <= new Date(filters.endDate));
        }

        // 计算分页
        this.pagination.totalPages = Math.ceil(filteredTransactions.length / this.pagination.pageSize);
        
        // 确保当前页码有效
        if (this.pagination.currentPage > this.pagination.totalPages) {
            this.pagination.currentPage = this.pagination.totalPages || 1;
        }

        // 获取当前页的数据
        const startIndex = (this.pagination.currentPage - 1) * this.pagination.pageSize;
        const endIndex = startIndex + this.pagination.pageSize;
        const currentPageData = filteredTransactions.slice(startIndex, endIndex);

        // 更新表格内容
        tbody.innerHTML = currentPageData.map(transaction => `
            <tr>
                <td>${transaction.id}</td>
                <td>${transaction.type}</td>
                <td><span class="text-${transaction.direction === '买入' ? 'success' : 'danger'}">${transaction.direction}</span></td>
                <td>¥${transaction.price}</td>
                <td>${transaction.volume} MWh</td>
                <td>¥${Number(transaction.amount).toLocaleString()}</td>
                <td><span class="badge bg-${transaction.statusClass}">${transaction.status}</span></td>
                <td>
                    <button type="button" class="btn btn-sm btn-light" onclick="window.tradingCharts.showTransactionDetail('${transaction.id}')">详情</button>
                    ${transaction.status === '待确认' ? `
                        <button type="button" class="btn btn-sm btn-danger" onclick="window.tradingCharts.cancelTransaction('${transaction.id}')">撤销</button>
                    ` : ''}
                </td>
            </tr>
        `).join('');

        // 更新分页控件
        this._updatePagination(filteredTransactions.length);
    }

    /**
     * 更新分页控件
     */
    _updatePagination(totalItems) {
        const paginationContainer = document.querySelector('.table-trading').closest('.card-body');
        
        // 如果没有分页容器，创建一个
        let paginationDiv = paginationContainer.querySelector('.pagination-container');
        if (!paginationDiv) {
            paginationDiv = document.createElement('div');
            paginationDiv.className = 'pagination-container d-flex justify-content-between align-items-center mt-3';
            paginationContainer.appendChild(paginationDiv);
        }

        // 生成分页HTML
        const totalPages = this.pagination.totalPages;
        const currentPage = this.pagination.currentPage;
        
        let paginationHtml = `
            <div class="pagination-info">
                显示 ${totalItems} 条记录中的 ${Math.min((currentPage - 1) * this.pagination.pageSize + 1, totalItems)} - 
                ${Math.min(currentPage * this.pagination.pageSize, totalItems)} 条
            </div>
            <ul class="pagination mb-0">
                <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
                    <a class="page-link" href="javascript:void(0)" onclick="window.tradingCharts.goToPage(1)">
                        <i class="ri-arrow-left-double-line"></i>
                    </a>
                </li>
                <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
                    <a class="page-link" href="javascript:void(0)" onclick="window.tradingCharts.goToPage(${currentPage - 1})">
                        <i class="ri-arrow-left-s-line"></i>
                    </a>
                </li>
        `;

        // 生成页码按钮
        let startPage = Math.max(1, currentPage - 2);
        let endPage = Math.min(totalPages, startPage + 4);
        startPage = Math.max(1, endPage - 4);

        for (let i = startPage; i <= endPage; i++) {
            paginationHtml += `
                <li class="page-item ${i === currentPage ? 'active' : ''}">
                    <a class="page-link" href="javascript:void(0)" onclick="window.tradingCharts.goToPage(${i})">${i}</a>
                </li>
            `;
        }

        paginationHtml += `
                <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
                    <a class="page-link" href="javascript:void(0)" onclick="window.tradingCharts.goToPage(${currentPage + 1})">
                        <i class="ri-arrow-right-s-line"></i>
                    </a>
                </li>
                <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
                    <a class="page-link" href="javascript:void(0)" onclick="window.tradingCharts.goToPage(${totalPages})">
                        <i class="ri-arrow-right-double-line"></i>
                    </a>
                </li>
            </ul>
        `;

        paginationDiv.innerHTML = paginationHtml;
    }

    /**
     * 跳转到指定页
     */
    goToPage(page) {
        if (page < 1 || page > this.pagination.totalPages) return;
        this.pagination.currentPage = page;
        this._updateTransactionTable();
    }

    /**
     * 显示交易记录筛选对话框
     */
    _showTransactionFilter() {
        // 创建筛选模态框
        const modalHtml = `
            <div class="modal fade" id="transactionFilterModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content bg-dark">
                        <div class="modal-header border-bottom border-secondary">
                            <h5 class="modal-title text-white">筛选交易记录</h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="mb-3">
                                <label class="form-label text-white">交易类型</label>
                                <select class="form-select bg-dark text-white border-secondary" id="filter-type">
                                    <option value="">全部</option>
                                    <option value="现货交易">现货交易</option>
                                    <option value="合约交易">合约交易</option>
                                    <option value="辅助服务">辅助服务</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label text-white">交易方向</label>
                                <select class="form-select bg-dark text-white border-secondary" id="filter-direction">
                                    <option value="">全部</option>
                                    <option value="买入">买入</option>
                                    <option value="卖出">卖出</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label text-white">交易状态</label>
                                <select class="form-select bg-dark text-white border-secondary" id="filter-status">
                                    <option value="">全部</option>
                                    <option value="已成交">已成交</option>
                                    <option value="待确认">待确认</option>
                                    <option value="已撤销">已撤销</option>
                                    <option value="执行中">执行中</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label text-white">时间范围</label>
                                <div class="input-group">
                                    <input type="date" class="form-control bg-dark text-white border-secondary" id="filter-start-date">
                                    <span class="input-group-text bg-dark text-white border-secondary">至</span>
                                    <input type="date" class="form-control bg-dark text-white border-secondary" id="filter-end-date">
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer border-top border-secondary">
                            <button type="button" class="btn btn-outline-light" onclick="window.tradingCharts.resetTransactionFilter()">重置</button>
                            <button type="button" class="btn btn-primary" onclick="window.tradingCharts.applyTransactionFilter()">应用筛选</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // 添加模态框到页面
        if (!document.getElementById('transactionFilterModal')) {
            document.body.insertAdjacentHTML('beforeend', modalHtml);
        }

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('transactionFilterModal'));
        modal.show();
    }

    /**
     * 应用交易记录筛选
     */
    applyTransactionFilter() {
        // 重置页码到第一页
        this.pagination.currentPage = 1;
        
        const filters = {
            type: document.getElementById('filter-type').value,
            direction: document.getElementById('filter-direction').value,
            status: document.getElementById('filter-status').value,
            startDate: document.getElementById('filter-start-date').value,
            endDate: document.getElementById('filter-end-date').value
        };

        this._updateTransactionTable(filters);

        // 关闭模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('transactionFilterModal'));
        modal.hide();

        // 显示提示
        this._showToast('筛选已应用', 'success');
    }

    /**
     * 重置交易记录筛选
     */
    resetTransactionFilter() {
        document.getElementById('filter-type').value = '';
        document.getElementById('filter-direction').value = '';
        document.getElementById('filter-status').value = '';
        document.getElementById('filter-start-date').value = '';
        document.getElementById('filter-end-date').value = '';

        this._updateTransactionTable();

        // 显示提示
        this._showToast('筛选已重置', 'success');
    }

    /**
     * 导出交易记录
     */
    _exportTransactions() {
        const data = this.transactions.map(t => ({
            '交易ID': t.id,
            '交易类型': t.type,
            '交易方向': t.direction,
            '价格(元/MWh)': t.price,
            '电量(MWh)': t.volume,
            '金额(元)': t.amount,
            '状态': t.status,
            '交易时间': new Date(t.timestamp).toLocaleString(),
            '策略': t.strategy
        }));

        // 创建工作簿
        const wb = XLSX.utils.book_new();
        const ws = XLSX.utils.json_to_sheet(data);

        // 设置列宽
        const colWidths = [
            { wch: 20 }, // 交易ID
            { wch: 12 }, // 交易类型
            { wch: 10 }, // 交易方向
            { wch: 15 }, // 价格
            { wch: 12 }, // 电量
            { wch: 15 }, // 金额
            { wch: 10 }, // 状态
            { wch: 20 }, // 交易时间
            { wch: 15 }  // 策略
        ];
        ws['!cols'] = colWidths;

        // 添加工作表到工作簿
        XLSX.utils.book_append_sheet(wb, ws, "交易记录");

        // 导出文件
        const fileName = `交易记录_${new Date().toISOString().split('T')[0]}.xlsx`;
        XLSX.writeFile(wb, fileName);

        // 显示提示
        this._showToast('交易记录已导出', 'success');
    }

    /**
     * 显示交易详情
     */
    showTransactionDetail(transactionId) {
        const transaction = this.transactions.find(t => t.id === transactionId);
        if (!transaction) return;

        // 创建详情模态框
        const modalHtml = `
            <div class="modal fade" id="transactionDetailModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">交易详情</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="mb-3">
                                <label class="text-muted">交易ID</label>
                                <div class="form-control">${transaction.id}</div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-6">
                                    <label class="text-muted">交易类型</label>
                                    <div class="form-control">${transaction.type}</div>
                                </div>
                                <div class="col-6">
                                    <label class="text-muted">交易方向</label>
                                    <div class="form-control">
                                        <span class="text-${transaction.direction === '买入' ? 'success' : 'danger'}">${transaction.direction}</span>
                                    </div>
                                </div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-6">
                                    <label class="text-muted">价格</label>
                                    <div class="form-control">¥${transaction.price}/MWh</div>
                                </div>
                                <div class="col-6">
                                    <label class="text-muted">电量</label>
                                    <div class="form-control">${transaction.volume} MWh</div>
                                </div>
                            </div>
                            <div class="mb-3">
                                <label class="text-muted">交易金额</label>
                                <div class="form-control">¥${Number(transaction.amount).toLocaleString()}</div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-6">
                                    <label class="text-muted">交易状态</label>
                                    <div class="form-control">
                                        <span class="badge bg-${transaction.statusClass}">${transaction.status}</span>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <label class="text-muted">交易时间</label>
                                    <div class="form-control">${new Date(transaction.timestamp).toLocaleString()}</div>
                                </div>
                            </div>
                            <div class="mb-3">
                                <label class="text-muted">执行策略</label>
                                <div class="form-control">${transaction.strategy}</div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-light" data-bs-dismiss="modal">关闭</button>
                            ${transaction.status === '待确认' ? `
                                <button type="button" class="btn btn-danger" onclick="window.tradingCharts.cancelTransaction('${transaction.id}')">撤销交易</button>
                            ` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;

        // 添加模态框到页面
        if (!document.getElementById('transactionDetailModal')) {
            document.body.insertAdjacentHTML('beforeend', modalHtml);
        } else {
            document.getElementById('transactionDetailModal').outerHTML = modalHtml;
        }

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('transactionDetailModal'));
        modal.show();
    }

    /**
     * 撤销交易
     */
    cancelTransaction(transactionId) {
        const transaction = this.transactions.find(t => t.id === transactionId);
        if (!transaction || transaction.status !== '待确认') return;

        // 更新交易状态
        transaction.status = '已撤销';
        transaction.statusClass = 'danger';

        // 更新表格显示
        this._updateTransactionTable();

        // 关闭详情模态框（如果打开的话）
        const detailModal = bootstrap.Modal.getInstance(document.getElementById('transactionDetailModal'));
        if (detailModal) {
            detailModal.hide();
        }

        // 显示提示
        this._showToast('交易已撤销', 'success');
    }
}

// 导出图表类
window.TradingCharts = TradingCharts; 


// 生成模拟数据
function generateChartData() {
    const data = [];
    let date = new Date(2024, 0, 1);
    let price = 480;
    
    for(let i = 0; i < 90; i++) {
        const open = price + (Math.random() - 0.5) * 10;
        const high = open + Math.random() * 8;
        const low = open - Math.random() * 8;
        const close = (high + low) / 2;
        const volume = Math.floor(Math.random() * 1000) + 500;
        
        data.push({
            x: new Date(date),
            y: [open, high, low, close],
            volume: volume
        });
        
        date.setDate(date.getDate() + 1);
        price = close;
    }
    return data;
}

// 初始化图表
const chartData = generateChartData();

// K线图配置
const candlestickOptions = {
    series: [{
        name: '电力价格',
        data: chartData
    }],
    chart: {
        type: 'candlestick',
        height: 400,
        toolbar: {
            show: true,
            tools: {
                download: false,
                selection: true,
                zoom: true,
                zoomin: true,
                zoomout: true,
                pan: true,
            }
        },
        background: '#2a3042',
        animations: {
            enabled: true,
            easing: 'easeinout',
            speed: 800
        }
    },
    plotOptions: {
        candlestick: {
            colors: {
                upward: '#26a69a',
                downward: '#ef5350'
            },
            wick: {
                useFillColor: true
            }
        }
    },
    title: {
        text: '电力现货价格走势',
        align: 'left',
        style: {
            fontSize: '16px',
            color: '#cccccc'
        }
    },
    xaxis: {
        type: 'datetime',
        labels: {
            style: {
                colors: '#ccc'
            },
            datetimeFormatter: {
                year: 'yyyy年',
                month: 'MM月',
                day: 'dd日',
                hour: 'HH:mm'
            }
        },
        title: {
            text: '交易时间',
            style: {
                color: '#cccccc'
            }
        },
        tooltip: {
            enabled: true
        }
    },
    yaxis: {
        tooltip: {
            enabled: true
        },
        labels: {
            style: {
                colors: '#ccc'
            },
            formatter: function (value) {
                return '¥' + value.toFixed(2);
            }
        },
        title: {
            text: '电力价格 (元/MWh)',
            style: {
                color: '#cccccc'
            }
        }
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
        theme: 'dark',
        custom: function({ seriesIndex, dataPointIndex, w }) {
            const o = w.globals.seriesCandleO[seriesIndex][dataPointIndex];
            const h = w.globals.seriesCandleH[seriesIndex][dataPointIndex];
            const l = w.globals.seriesCandleL[seriesIndex][dataPointIndex];
            const c = w.globals.seriesCandleC[seriesIndex][dataPointIndex];
            const date = new Date(w.globals.seriesX[seriesIndex][dataPointIndex]);
            
            return `
            <div class="p-2">
                <div class="mb-2">${date.toLocaleString('zh-CN')}</div>
                <div class="d-flex justify-content-between mb-1">
                    <span>开盘价：</span>
                    <span class="ms-2">¥${o.toFixed(2)}</span>
                </div>
                <div class="d-flex justify-content-between mb-1">
                    <span>最高价：</span>
                    <span class="ms-2">¥${h.toFixed(2)}</span>
                </div>
                <div class="d-flex justify-content-between mb-1">
                    <span>最低价：</span>
                    <span class="ms-2">¥${l.toFixed(2)}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>收盘价：</span>
                    <span class="ms-2">¥${c.toFixed(2)}</span>
                </div>
            </div>`;
        }
    },
    annotations: {
        yaxis: [{
            y: 480,
            borderColor: '#ffbf53',
            label: {
                text: '基准价格',
                style: {
                    color: '#fff',
                    background: '#ffbf53'
                }
            }
        }]
    }
};

// 成交量图配置
const volumeOptions = {
    series: [{
        name: '成交电量',
        data: chartData.map(item => ({
            x: item.x,
            y: item.volume
        }))
    }],
    chart: {
        height: 100,
        type: 'bar',
        brush: {
            enabled: true,
            target: 'tradingChart'
        },
        background: '#2a3042'
    },
    plotOptions: {
        bar: {
            colors: {
                ranges: [{
                    from: -1000000,
                    to: 0,
                    color: '#ef5350'
                }, {
                    from: 0,
                    to: 1000000,
                    color: '#26a69a'
                }]
            }
        }
    },
    dataLabels: {
        enabled: false
    },
    stroke: {
        width: 0
    },
    xaxis: {
        type: 'datetime',
        labels: {
            style: {
                colors: '#ccc'
            },
            datetimeFormatter: {
                year: 'yyyy年',
                month: 'MM月',
                day: 'dd日',
                hour: 'HH:mm'
            }
        }
    },
    yaxis: {
        labels: {
            style: {
                colors: '#ccc'
            },
            formatter: function (value) {
                return value.toFixed(0) + ' MWh';
            }
        },
        title: {
            text: '成交电量 (MWh)',
            style: {
                color: '#cccccc'
            }
        }
    },
    grid: {
        borderColor: '#404040'
    },
    tooltip: {
        theme: 'dark',
        y: {
            formatter: function(value) {
                return value.toFixed(0) + ' MWh';
            }
        }
    }
};

// 创建图表实例
const tradingChart = new ApexCharts(document.querySelector("#tradingChart"), candlestickOptions);
const volumeChart = new ApexCharts(document.querySelector("#volumeChart"), volumeOptions);

// 渲染图表
tradingChart.render();
volumeChart.render();

// 更新图表周期
function updateChartPeriod(period) {
    // 根据选择的时间周期过滤数据
    let filteredData = [...chartData];
    const now = new Date();
    let periodText = '';
    
    switch(period) {
        case '1D':
            filteredData = chartData.slice(-24);
            periodText = '24小时';
            break;
        case '7D':
            filteredData = chartData.slice(-7 * 24);
            periodText = '7天';
            break;
        case '1M':
            filteredData = chartData.slice(-30 * 24);
            periodText = '30天';
            break;
        case 'ALL':
            periodText = '全部';
            break;
    }
    
    // 更新图表标题
    tradingChart.updateOptions({
        title: {
            text: `电力现货价格走势 (${periodText})`
        }
    });
    
    // 更新图表数据
    tradingChart.updateSeries([{
        data: filteredData
    }]);
    
    volumeChart.updateSeries([{
        data: filteredData.map(item => ({
            x: item.x,
            y: item.volume
        }))
    }]);
}

// 添加图例说明
document.addEventListener('DOMContentLoaded', function() {
    const legendContainer = document.createElement('div');
    legendContainer.className = 'trading-chart-legend mt-3 p-3 border rounded';
    legendContainer.innerHTML = `
        <h6 class="mb-3">图表说明：</h6>
        <div class="row">
            <div class="col-md-6">
                <p class="mb-2"><i class="fas fa-square text-success me-2"></i>绿色K线：收盘价高于开盘价（上涨）</p>
                <p class="mb-2"><i class="fas fa-square text-danger me-2"></i>红色K线：收盘价低于开盘价（下跌）</p>
            </div>
            <div class="col-md-6">
                <p class="mb-2"><i class="fas fa-arrows-alt-v me-2"></i>K线上下影线：当日最高价和最低价范围</p>
                <p class="mb-2"><i class="fas fa-chart-bar me-2"></i>下方柱状图：每个时间点的成交电量</p>
            </div>
        </div>
    `;
    document.querySelector('.trading-chart-container').appendChild(legendContainer);
});