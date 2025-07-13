// 日期格式化工具函数
function formatDateTime(dateStr) {
    const date = new Date(dateStr);
    return date.toISOString().slice(0, 16).replace('T', ' ');
}

// 优化的筛选功能
function applyFilter() {
    const type = document.getElementById('filter-type').value;
    const status = document.getElementById('filter-status').value;
    const startTime = document.getElementById('filter-start-time').value;
    const endTime = document.getElementById('filter-end-time').value;
    const searchField = document.getElementById('filter-field').value;
    const keyword = document.getElementById('filter-keyword').value.toLowerCase();

    // 验证时间范围
    if (startTime && endTime && new Date(startTime) > new Date(endTime)) {
        showToast('开始时间不能晚于结束时间', 'danger');
        return;
    }

    // 转换时间格式
    const startDateTime = startTime ? formatDateTime(startTime) : '';
    const endDateTime = endTime ? formatDateTime(endTime) : '';

    console.log('Filter criteria:', {
        type,
        status,
        startDateTime,
        endDateTime,
        searchField,
        keyword
    });

    const rows = document.querySelectorAll('#business-records-table tbody tr');
    let visibleCount = 0;
    
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        const rowType = cells[1].textContent.trim();
        const rowStatus = cells[6].querySelector('.badge').textContent.trim();
        const rowTime = cells[4].textContent.trim();
        
        // 构建搜索文本
        let searchText;
        if (searchField === 'all') {
            searchText = [
                cells[0].textContent.trim(), // 记录ID
                cells[2].textContent.trim(), // 执行内容
                cells[3].textContent.trim()  // 目标对象
            ].join(' ').toLowerCase();
        } else {
            const cellIndex = {
                'recordId': 0,
                'content': 2,
                'target': 3
            }[searchField];
            searchText = cells[cellIndex].textContent.trim().toLowerCase();
        }

        // 调试输出
        console.log('Row data:', {
            rowType,
            rowStatus,
            rowTime,
            searchText
        });

        let show = true;

        // 应用筛选条件
        if (type && rowType !== type) {
            console.log('Type mismatch:', rowType, '!==', type);
            show = false;
        }
        if (status && rowStatus !== status) {
            console.log('Status mismatch:', rowStatus, '!==', status);
            show = false;
        }
        if (startDateTime && rowTime < startDateTime) {
            console.log('Start time mismatch:', rowTime, '<', startDateTime);
            show = false;
        }
        if (endDateTime && rowTime > endDateTime) {
            console.log('End time mismatch:', rowTime, '>', endDateTime);
            show = false;
        }
        if (keyword && !searchText.includes(keyword)) {
            console.log('Keyword not found:', keyword, 'in', searchText);
            show = false;
        }

        console.log('Show row:', show);

        row.style.display = show ? '' : 'none';
        if (show) visibleCount++;
    });

    // 更新表格显示状态
    const table = document.getElementById('business-records-table');
    const noDataMsg = document.getElementById('no-records-message');
    
    if (visibleCount === 0) {
        table.classList.add('d-none');
        if (!noDataMsg) {
            const msg = document.createElement('div');
            msg.id = 'no-records-message';
            msg.className = 'alert alert-info text-center';
            msg.innerHTML = `
                <i class="fas fa-info-circle me-2"></i>
                没有找到匹配的记录
                <div class="small text-muted mt-2">
                    当前筛选条件：
                    ${type ? `<br>业务类型: ${type}` : ''}
                    ${status ? `<br>状态: ${status}` : ''}
                    ${startDateTime ? `<br>开始时间: ${startDateTime}` : ''}
                    ${endDateTime ? `<br>结束时间: ${endDateTime}` : ''}
                    ${keyword ? `<br>关键词: ${keyword}` : ''}
                </div>
            `;
            table.parentNode.insertBefore(msg, table);
        } else {
            noDataMsg.classList.remove('d-none');
        }
    } else {
        table.classList.remove('d-none');
        if (noDataMsg) noDataMsg.classList.add('d-none');
    }

    // 关闭筛选模态框
    const filterModal = bootstrap.Modal.getInstance(document.getElementById('filterModal'));
    filterModal.hide();

    // 显示筛选结果提示
    showToast(`筛选完成，共找到 ${visibleCount} 条记录`, visibleCount > 0 ? 'success' : 'warning');
}

// 初始化筛选功能
function initializeFilter() {
    // 获取元素
    const typeSelect = document.getElementById('filter-type');
    const statusSelect = document.getElementById('filter-status');
    const startTimeInput = document.getElementById('filter-start-time');
    const endTimeInput = document.getElementById('filter-end-time');

    // 检查元素是否存在
    if (!typeSelect || !statusSelect || !startTimeInput || !endTimeInput) {
        console.warn('Filter elements not found, skipping filter initialization');
        return;
    }

    try {
    // 动态加载业务类型
    const types = new Set();
    document.querySelectorAll('#business-records-table tbody tr').forEach(row => {
            if (row.cells[1]) {
        types.add(row.cells[1].textContent.trim());
            }
    });
    typeSelect.innerHTML = '<option value="">全部</option>' + 
        Array.from(types).sort().map(type => `<option value="${type}">${type}</option>`).join('');

    // 动态加载状态
    const statuses = new Set();
    document.querySelectorAll('#business-records-table tbody tr').forEach(row => {
            const badge = row.cells[6]?.querySelector('.badge');
            if (badge) {
                statuses.add(badge.textContent.trim());
            }
    });
    statusSelect.innerHTML = '<option value="">全部</option>' + 
        Array.from(statuses).sort().map(status => `<option value="${status}">${status}</option>`).join('');

    // 设置默认时间范围（最近一周）
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        startTimeInput.value = weekAgo.toISOString().slice(0, 16);
        endTimeInput.value = now.toISOString().slice(0, 16);

    // 添加调试信息
        console.log('Filter initialized with types:', Array.from(types));
        console.log('Filter initialized with statuses:', Array.from(statuses));
    } catch (error) {
        console.error('Error initializing filter:', error);
    }
}

// 重置筛选条件
function resetFilter() {
    document.getElementById('filter-form').reset();
    initializeFilter(); // 重新初始化时间范围
    
    // 显示所有记录
    document.querySelectorAll('#business-records-table tbody tr').forEach(row => {
        row.style.display = '';
    });
    
    // 移除无数据提示
    const noDataMsg = document.getElementById('no-records-message');
    if (noDataMsg) noDataMsg.remove();
    
    // 显示表格
    document.getElementById('business-records-table').classList.remove('d-none');
    
    showToast('筛选条件已重置', 'success');
}

// 绑定事件
document.addEventListener('DOMContentLoaded', function() {
    // 初始化筛选器
    initializeFilter();

    // 绑定筛选按钮事件
    document.getElementById('apply-filter').addEventListener('click', applyFilter);
    document.getElementById('reset-filter').addEventListener('click', resetFilter);
    
    // 添加回车键提交功能
    document.getElementById('filter-keyword').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            applyFilter();
        }
    });

    // 添加时间输入验证
    const startTimeInput = document.getElementById('filter-start-time');
    const endTimeInput = document.getElementById('filter-end-time');

    startTimeInput.addEventListener('change', function() {
        if (endTimeInput.value && this.value > endTimeInput.value) {
            showToast('开始时间不能晚于结束时间', 'danger');
            this.value = endTimeInput.value;
        }
    });

    endTimeInput.addEventListener('change', function() {
        if (startTimeInput.value && this.value < startTimeInput.value) {
            showToast('结束时间不能早于开始时间', 'danger');
            this.value = startTimeInput.value;
        }
    });
});

// 业务处理模块
const BusinessProcessor = {
    // 配置参数
    config: {
        // 预测任务相关配置
        prediction: {
            errorThreshold: 0.05, // 预测误差阈值 5%
            updateInterval: 900000, // 更新间隔 15分钟
            confidenceLevel: 0.95, // 置信水平
            maxRetries: 3 // 最大重试次数
        },
        // 储能相关配置
        storage: {
            minSoc: 0.2, // 最小充电状态
            maxSoc: 0.8, // 最大充电状态
            chargingEfficiency: 0.95 // 充放电效率
        },
        // 告警相关配置
        alert: {
            levels: {
                info: 'info',
                warning: 'warning',
                error: 'error',
                critical: 'critical'
            },
            retentionDays: 7 // 告警数据保留天数
        },
        // 缓存配置
        cache: {
            strategy: 'LRU',
            maxSize: '10GB',
            retentionDays: 7
        },
        // 调度相关配置
        dispatch: {
            minPower: 0, // 最小功率
            maxPower: 1000, // 最大功率 (MW)
            rampRate: 10, // 爬坡率 (MW/min)
            responseTime: 5000 // 响应时间限制 (ms)
        },
        // 交易相关配置
        trading: {
            minVolume: 1, // 最小交易量 (MWh)
            maxVolume: 1000, // 最大交易量 (MWh)
            priceLimit: 1000, // 价格上限 (元/MWh)
            settlementTime: 900000 // 结算时间 (ms)
        },
        // 检修相关配置
        maintenance: {
            maxDuration: 168, // 最大检修时长 (小时)
            warningThreshold: 48, // 预警时间阈值 (小时)
            priorityLevels: ['low', 'medium', 'high', 'urgent']
        },
        // 应急响应配置
        emergency: {
            responseTimeout: 300000, // 响应超时时间 (ms)
            escalationLevels: ['info', 'warning', 'critical', 'emergency'],
            autoEscalation: true // 是否自动升级
        }
    },

    // 风光发电预测业务处理
    renewableEnergyProcessor: {
        // 预测任务调度
        schedulePrediction: async function(deviceId, weatherData, deviceStatus) {
            try {
                console.log('开始调度预测任务:', deviceId);
                
                // 数据校验
                if (!this._validateInputData(weatherData, deviceStatus)) {
                    throw new Error('输入数据验证失败');
                }

                // 获取历史数据
                const historicalData = await this._getHistoricalData(deviceId);
                
                // 执行预测
                const predictionResult = await this._executePrediction({
                    deviceId,
                    weatherData,
                    deviceStatus,
                    historicalData
                });

                // 结果校验
                const validationResult = this._validatePredictionResult(predictionResult);
                if (!validationResult.isValid) {
                    BusinessProcessor.alertProcessor.createAlert({
                        level: 'warning',
                        message: `预测结果异常: ${validationResult.reason}`,
                        deviceId
                    });
                }

                return predictionResult;
            } catch (error) {
                console.error('预测任务调度失败:', error);
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `预测任务调度失败: ${error.message}`,
                    deviceId
                });
                throw error;
            }
        },

        // 结果校验与修正
        validateAndCorrect: function(predictionResult, actualData) {
            const error = Math.abs(predictionResult - actualData) / actualData;
            
            if (error > BusinessProcessor.config.prediction.errorThreshold) {
                // 触发告警
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'warning',
                    message: `预测误差超过阈值: ${(error * 100).toFixed(2)}%`,
                    data: { predictionResult, actualData }
                });

                // 执行修正
                return this._correctPrediction(predictionResult, actualData);
            }

            return predictionResult;
        },

        // 内部方法：数据验证
        _validateInputData: function(weatherData, deviceStatus) {
            return weatherData && deviceStatus && 
                   weatherData.temperature !== undefined &&
                   weatherData.windSpeed !== undefined &&
                   deviceStatus.operational !== undefined;
        },

        // 内部方法：获取历史数据
        _getHistoricalData: async function(deviceId) {
            // 实现从缓存或数据库获取历史数据的逻辑
            return [];
        },

        // 内部方法：执行预测
        _executePrediction: async function(data) {
            // 实现预测算法的逻辑
            return 0;
        },

        // 内部方法：验证预测结果
        _validatePredictionResult: function(result) {
            return {
                isValid: result !== undefined && !isNaN(result),
                reason: result === undefined ? '预测结果为空' : 
                        isNaN(result) ? '预测结果非数字' : ''
            };
        },

        // 内部方法：修正预测结果
        _correctPrediction: function(prediction, actual) {
            // 实现预测结果修正的逻辑
            return prediction * 0.9 + actual * 0.1;
        },

        // 初始化图表
        initPredictionChart: function() {
            const options = {
                series: [{
                    name: '预测发电量',
                    data: []
                }, {
                    name: '实际发电量',
                    data: []
                }],
                chart: {
                    type: 'line',
                    height: 300
                },
                xaxis: {
                    type: 'datetime'
                },
                yaxis: {
                    title: {
                        text: '发电量 (MWh)'
                    }
                }
            };

            const chart = new ApexCharts(document.querySelector("#renewable-prediction-chart"), options);
            chart.render();
            return chart;
        },

        // 更新历史记录表格
        updateHistoryTable: function(data) {
            const tableBody = document.querySelector('#renewable-history-table tbody');
            tableBody.innerHTML = data.map(record => `
                <tr>
                    <td>${record.taskId}</td>
                    <td>${record.device}</td>
                    <td>${record.predictionTime}</td>
                    <td>${record.predictedValue}</td>
                    <td>${record.actualValue || '-'}</td>
                    <td>${record.errorRate ? record.errorRate.toFixed(2) + '%' : '-'}</td>
                    <td><span class="badge bg-${record.status === '完成' ? 'success' : 'warning'}">${record.status}</span></td>
                </tr>
            `).join('');
        }
    },

    // 能源交易业务处理
    energyTradingProcessor: {
        // 市场数据接入
        fetchMarketData: async function() {
            try {
                // 实现从市场API获取数据的逻辑
                const marketData = await this._fetchFromMarketAPI();
                
                // 数据验证和处理
                if (this._validateMarketData(marketData)) {
                    // 缓存数据
                    await this._cacheMarketData(marketData);
                    return marketData;
                }
                
                throw new Error('市场数据验证失败');
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `市场数据获取失败: ${error.message}`
                });
                throw error;
            }
        },

        // 生成交易策略
        generateTradingStrategy: function(predictionData, marketData) {
            try {
                // 分析供需情况
                const supplyDemandAnalysis = this._analyzeSupplyDemand(predictionData);
                
                // 分析市场价格
                const priceAnalysis = this._analyzePricePattern(marketData);
                
                // 生成交易建议
                return this._generateTradeRecommendations(supplyDemandAnalysis, priceAnalysis);
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `交易策略生成失败: ${error.message}`
                });
                throw error;
            }
        },

        // 内部方法：从市场API获取数据
        _fetchFromMarketAPI: async function() {
            // 实现市场API调用逻辑
            return {};
        },

        // 内部方法：验证市场数据
        _validateMarketData: function(data) {
            return data && data.prices && Array.isArray(data.prices);
        },

        // 内部方法：缓存市场数据
        _cacheMarketData: async function(data) {
            // 实现数据缓存逻辑
        },

        // 内部方法：分析供需情况
        _analyzeSupplyDemand: function(predictionData) {
            // 实现供需分析逻辑
            return {};
        },

        // 内部方法：分析价格模式
        _analyzePricePattern: function(marketData) {
            // 实现价格分析逻辑
            return {};
        },

        // 内部方法：生成交易建议
        _generateTradeRecommendations: function(supplyDemand, priceAnalysis) {
            // 实现交易建议生成逻辑
            return {};
        },

        // 初始化市场分析图表
        initMarketChart: function() {
            const options = {
                series: [{
                    name: '市场价格',
                    data: []
                }, {
                    name: '交易量',
                    data: []
                }],
                chart: {
                    type: 'line',
                    height: 300
                },
                xaxis: {
                    type: 'datetime'
                },
                yaxis: [{
                    title: {
                        text: '价格 (元/MWh)'
                    }
                }, {
                    opposite: true,
                    title: {
                        text: '交易量 (MWh)'
                    }
                }]
            };

            const chart = new ApexCharts(document.querySelector("#market-analysis-chart"), options);
            chart.render();
            return chart;
        },

        // 更新交易记录表格
        updateTradingHistory: function(data) {
            const tableBody = document.querySelector('#trading-history-table tbody');
            tableBody.innerHTML = data.map(record => `
                <tr>
                    <td>${record.tradeId}</td>
                    <td>${record.type}</td>
                    <td>${record.direction}</td>
                    <td>${record.volume} MWh</td>
                    <td>${record.price}</td>
                    <td>${record.executeTime}</td>
                    <td><span class="badge bg-${record.status === 'completed' ? 'success' : 'warning'}">${record.status}</span></td>
                    <td>
                        <button class="btn btn-sm btn-primary">详情</button>
                        ${record.status !== 'completed' ? '<button class="btn btn-sm btn-danger">取消</button>' : ''}
                    </td>
                </tr>
            `).join('');
        }
    },

    // 电力负荷预测业务处理
    loadPredictionProcessor: {
        // 负荷趋势分析
        analyzeTrend: async function(historicalData, weatherForecast) {
            try {
                // 数据预处理
                const processedData = this._preprocessData(historicalData);
                
                // 趋势分析
                const trend = this._calculateTrend(processedData, weatherForecast);
                
                // 结果验证
                if (this._validateTrendResult(trend)) {
                    return trend;
                }
                
                throw new Error('趋势分析结果无效');
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `负荷趋势分析失败: ${error.message}`
                });
                throw error;
            }
        },

        // 供需匹配分析
        analyzeSupplyDemandMatch: function(supplyPrediction, demandPrediction) {
            try {
                // 计算供需差异
                const gap = this._calculateSupplyDemandGap(supplyPrediction, demandPrediction);
                
                // 生成调整建议
                return this._generateAdjustmentSuggestions(gap);
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `供需匹配分析失败: ${error.message}`
                });
                throw error;
            }
        },

        // 内部方法：数据预处理
        _preprocessData: function(data) {
            // 实现数据预处理逻辑
            return data;
        },

        // 内部方法：计算趋势
        _calculateTrend: function(data, weather) {
            // 实现趋势计算逻辑
            return {};
        },

        // 内部方法：验证趋势结果
        _validateTrendResult: function(trend) {
            return trend && typeof trend === 'object';
        },

        // 内部方法：计算供需差异
        _calculateSupplyDemandGap: function(supply, demand) {
            // 实现供需差异计算逻辑
            return {};
        },

        // 内部方法：生成调整建议
        _generateAdjustmentSuggestions: function(gap) {
            // 实现调整建议生成逻辑
            return {};
        },

        // 初始化负荷趋势图表
        initLoadTrendChart: function() {
            const options = {
                series: [{
                    name: '预测负荷',
                    data: []
                }, {
                    name: '实际负荷',
                    data: []
                }],
                chart: {
                    type: 'line',
                    height: 300
                },
                xaxis: {
                    type: 'datetime'
                },
                yaxis: {
                    title: {
                        text: '负荷 (MW)'
                    }
                }
            };

            const chart = new ApexCharts(document.querySelector("#load-trend-chart"), options);
            chart.render();
            return chart;
        },

        // 更新预测结果表格
        updatePredictionTable: function(data) {
            const tableBody = document.querySelector('#load-prediction-table tbody');
            tableBody.innerHTML = data.map(record => `
                <tr>
                    <td>${record.predictionId}</td>
                    <td>${record.area}</td>
                    <td>${record.timeRange}</td>
                    <td>${record.predictedLoad}</td>
                    <td>${record.actualLoad || '-'}</td>
                    <td>${record.errorRate ? record.errorRate.toFixed(2) + '%' : '-'}</td>
                    <td><span class="badge bg-${record.status === '完成' ? 'success' : 'warning'}">${record.status}</span></td>
                </tr>
            `).join('');
        }
    },

    // 充电量预测业务处理
    chargingPredictionProcessor: {
        // 充电需求预测
        predictChargingDemand: async function(historicalData, timeSlot) {
            try {
                // 数据验证
                if (!this._validateChargingData(historicalData)) {
                    throw new Error('充电历史数据无效');
                }

                // 执行预测
                const prediction = await this._executeChargingPrediction(historicalData, timeSlot);
                
                // 结果验证
                if (this._validatePredictionResult(prediction)) {
                    return prediction;
                }
                
                throw new Error('充电需求预测结果无效');
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `充电需求预测失败: ${error.message}`
                });
                throw error;
            }
        },

        // 资源分配优化
        optimizeResourceAllocation: function(prediction, availableResources) {
            try {
                // 计算最优分配方案
                const allocation = this._calculateOptimalAllocation(prediction, availableResources);
                
                // 生成执行计划
                return this._generateExecutionPlan(allocation);
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `资源分配优化失败: ${error.message}`
                });
                throw error;
            }
        },

        // 内部方法：验证充电数据
        _validateChargingData: function(data) {
            return data && Array.isArray(data) && data.length > 0;
        },

        // 内部方法：执行充电预测
        _executeChargingPrediction: async function(data, timeSlot) {
            // 实现充电预测逻辑
            return {};
        },

        // 内部方法：验证预测结果
        _validatePredictionResult: function(result) {
            return result && typeof result === 'object';
        },

        // 内部方法：计算最优分配
        _calculateOptimalAllocation: function(prediction, resources) {
            // 实现资源分配优化逻辑
            return {};
        },

        // 内部方法：生成执行计划
        _generateExecutionPlan: function(allocation) {
            // 实现执行计划生成逻辑
            return {};
        },

        // 初始化充电需求图表
        initChargingDemandChart: function() {
            const options = {
                series: [{
                    name: '预测需求',
                    data: []
                }, {
                    name: '实际需求',
                    data: []
                }],
                chart: {
                    type: 'line',
                    height: 300
                },
                xaxis: {
                    type: 'datetime'
                },
                yaxis: {
                    title: {
                        text: '充电需求 (MWh)'
                    }
                }
            };

            const chart = new ApexCharts(document.querySelector("#charging-demand-chart"), options);
            chart.render();
            return chart;
        },

        // 更新充电站状态表格
        updateStationStatus: function(data) {
            const tableBody = document.querySelector('#charging-status-table tbody');
            tableBody.innerHTML = data.map(record => `
                <tr>
                    <td>${record.chargerId}</td>
                    <td><span class="badge bg-${record.status === '空闲' ? 'success' : 'warning'}">${record.status}</span></td>
                    <td>${record.currentPower} kW</td>
                    <td>${record.totalCharged} kWh</td>
                    <td>${record.estimatedEndTime || '-'}</td>
                    <td>
                        <button class="btn btn-sm btn-primary">详情</button>
                        ${record.status !== '空闲' ? '<button class="btn btn-sm btn-danger">终止</button>' : ''}
                    </td>
                </tr>
            `).join('');
        }
    },

    // 电力调度处理器
    dispatchProcessor: {
        // 下发调度指令
        sendDispatchCommand: async function(command) {
            try {
                // 验证调度指令
                if (!this._validateDispatchCommand(command)) {
                    throw new Error('调度指令验证失败');
                }

                // 检查设备状态
                const deviceStatus = await this._checkDeviceStatus(command.targetId);
                if (!deviceStatus.available) {
                    throw new Error(`设备 ${command.targetId} 不可用: ${deviceStatus.reason}`);
                }

                // 执行调度
                const result = await this._executeDispatch(command);
                
                // 记录调度结果
                await this._recordDispatchResult(result);

                return result;
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `调度指令执行失败: ${error.message}`,
                    data: command
                });
                throw error;
            }
        },

        // 取消调度指令
        cancelDispatchCommand: async function(commandId) {
            try {
                const command = await this._getDispatchCommand(commandId);
                if (!command) {
                    throw new Error('调度指令不存在');
                }

                // 执行取消操作
                await this._executeCancellation(command);
                
                // 更新记录
                await this._updateDispatchRecord(commandId, 'cancelled');

                return { success: true, message: '调度指令已取消' };
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `取消调度指令失败: ${error.message}`,
                    commandId
                });
                throw error;
            }
        },

        // 内部方法：验证调度指令
        _validateDispatchCommand: function(command) {
            return command && 
                   command.type && 
                   command.targetId && 
                   typeof command.value === 'number' &&
                   command.value >= BusinessProcessor.config.dispatch.minPower &&
                   command.value <= BusinessProcessor.config.dispatch.maxPower;
        },

        // 内部方法：检查设备状态
        _checkDeviceStatus: async function(deviceId) {
            // 实现设备状态检查逻辑
            return { available: true };
        },

        // 内部方法：执行调度
        _executeDispatch: async function(command) {
            // 实现调度执行逻辑
            return {
                commandId: `DSP${Date.now()}`,
                status: 'executing',
                startTime: new Date()
            };
        },

        // 内部方法：记录调度结果
        _recordDispatchResult: async function(result) {
            // 实现结果记录逻辑
        },

        // 更新调度监控图表
        updateDispatchMonitor: function(data) {
            const chart = document.querySelector('#dispatch-monitor-chart');
            if (chart && chart.chart) {
                chart.chart.updateSeries([{
                    name: '实际功率',
                    data: data.actualPower
                }, {
                    name: '目标功率',
                    data: data.targetPower
                }]);
            }
        },

        // 更新调度历史记录
        updateDispatchHistory: function(records) {
            const tableBody = document.querySelector('#dispatch-history-table tbody');
            if (tableBody) {
                tableBody.innerHTML = records.map(record => `
                    <tr>
                        <td>${record.commandId}</td>
                        <td>${record.type}</td>
                        <td>${record.target}</td>
                        <td>${record.value} MW</td>
                        <td>${record.executeTime}</td>
                        <td><span class="badge bg-${record.status === 'completed' ? 'success' : 'warning'}">${record.status}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary">详情</button>
                            ${record.status !== 'completed' ? '<button class="btn btn-sm btn-danger">取消</button>' : ''}
                        </td>
                    </tr>
                `).join('');
            }
        }
    },

    // 能源交易处理器
    tradingProcessor: {
        // 执行交易
        executeTrade: async function(trade) {
            try {
                // 验证交易
                if (!this._validateTrade(trade)) {
                    throw new Error('交易参数验证失败');
                }

                // 检查市场状态
                const marketStatus = await this._checkMarketStatus();
                if (!marketStatus.isTrading) {
                    throw new Error('市场当前不可交易');
                }

                // 执行交易
                const result = await this._executeTransaction(trade);
                
                // 记录交易结果
                await this._recordTradeResult(result);

                return result;
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `交易执行失败: ${error.message}`,
                    data: trade
                });
                throw error;
            }
        },

        // 取消交易
        cancelTrade: async function(tradeId) {
            try {
                const trade = await this._getTrade(tradeId);
                if (!trade) {
                    throw new Error('交易不存在');
                }

                // 执行取消操作
                await this._executeCancellation(trade);
                
                // 更新记录
                await this._updateTradeRecord(tradeId, 'cancelled');

                return { success: true, message: '交易已取消' };
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `取消交易失败: ${error.message}`,
                    tradeId
                });
                throw error;
            }
        },

        // 内部方法：验证交易
        _validateTrade: function(trade) {
            return trade && 
                   trade.type && 
                   trade.direction &&
                   typeof trade.volume === 'number' &&
                   typeof trade.price === 'number' &&
                   trade.volume >= BusinessProcessor.config.trading.minVolume &&
                   trade.volume <= BusinessProcessor.config.trading.maxVolume &&
                   trade.price <= BusinessProcessor.config.trading.priceLimit;
        },

        // 内部方法：检查市场状态
        _checkMarketStatus: async function() {
            // 实现市场状态检查逻辑
            return { isTrading: true };
        },

        // 内部方法：执行交易
        _executeTransaction: async function(trade) {
            // 实现交易执行逻辑
            return {
                tradeId: `TRD${Date.now()}`,
                status: 'executing',
                timestamp: new Date()
            };
        },

        // 更新交易监控图表
        updateTradingMonitor: function(data) {
            const chart = document.querySelector('#trading-monitor-chart');
            if (chart && chart.chart) {
                chart.chart.updateSeries([{
                    name: '交易量',
                    data: data.volume
                }, {
                    name: '价格',
                    data: data.price
                }]);
            }
        },

        // 更新交易历史记录
        updateTradingHistory: function(records) {
            const tableBody = document.querySelector('#trading-history-table tbody');
            if (tableBody) {
                tableBody.innerHTML = records.map(record => `
                    <tr>
                        <td>${record.tradeId}</td>
                        <td>${record.type}</td>
                        <td>${record.direction}</td>
                        <td>${record.volume} MWh</td>
                        <td>${record.price}</td>
                        <td>${record.executeTime}</td>
                        <td><span class="badge bg-${record.status === 'completed' ? 'success' : 'warning'}">${record.status}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary">详情</button>
                            ${record.status !== 'completed' ? '<button class="btn btn-sm btn-danger">取消</button>' : ''}
                        </td>
                    </tr>
                `).join('');
            }
        }
    },

    // 设备检修处理器
    maintenanceProcessor: {
        // 创建检修任务
        createMaintenanceTask: async function(task) {
            try {
                // 验证任务
                if (!this._validateMaintenanceTask(task)) {
                    throw new Error('检修任务参数验证失败');
                }

                // 检查设备可用性
                const deviceStatus = await this._checkDeviceAvailability(task.deviceId);
                if (!deviceStatus.available) {
                    throw new Error(`设备 ${task.deviceId} 不可进行检修: ${deviceStatus.reason}`);
                }

                // 创建任务
                const result = await this._createTask(task);
                
                // 记录任务
                await this._recordMaintenanceTask(result);

                return result;
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `创建检修任务失败: ${error.message}`,
                    data: task
                });
                throw error;
            }
        },

        // 更新检修任务状态
        updateTaskStatus: async function(taskId, status, progress) {
            try {
                const task = await this._getMaintenanceTask(taskId);
                if (!task) {
                    throw new Error('检修任务不存在');
                }

                // 更新状态
                await this._updateTaskStatus(taskId, status, progress);
                
                // 记录更新
                await this._recordStatusUpdate(taskId, status, progress);

                return { success: true, message: '任务状态已更新' };
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `更新检修任务状态失败: ${error.message}`,
                    taskId
                });
                throw error;
            }
        },

        // 内部方法：验证检修任务
        _validateMaintenanceTask: function(task) {
            return task && 
                   task.deviceId && 
                   task.type &&
                   task.content &&
                   typeof task.duration === 'number' &&
                   task.duration <= BusinessProcessor.config.maintenance.maxDuration;
        },

        // 内部方法：检查设备可用性
        _checkDeviceAvailability: async function(deviceId) {
            // 实现设备可用性检查逻辑
            return { available: true };
        },

        // 内部方法：创建任务
        _createTask: async function(task) {
            // 实现任务创建逻辑
            return {
                taskId: `MNT${Date.now()}`,
                status: 'scheduled',
                createTime: new Date()
            };
        },

        // 更新检修进度图表
        updateMaintenanceProgress: function(data) {
            const chart = document.querySelector('#maintenance-progress-chart');
            if (chart && chart.chart) {
                chart.chart.updateSeries([{
                    name: '完成率',
                    data: data.progress
                }]);
            }
        },

        // 更新检修历史记录
        updateMaintenanceHistory: function(records) {
            const tableBody = document.querySelector('#maintenance-history-table tbody');
            if (tableBody) {
                tableBody.innerHTML = records.map(record => `
                    <tr>
                        <td>${record.taskId}</td>
                        <td>${record.device}</td>
                        <td>${record.type}</td>
                        <td>${record.planTime}</td>
                        <td>${record.duration}小时</td>
                        <td><span class="badge bg-${record.status === 'completed' ? 'success' : 'warning'}">${record.status}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary">详情</button>
                            ${record.status !== 'completed' ? '<button class="btn btn-sm btn-warning">更新</button>' : ''}
                        </td>
                    </tr>
                `).join('');
            }
        }
    },

    // 应急响应处理器
    emergencyProcessor: {
        // 启动应急响应
        startEmergencyResponse: async function(event) {
            try {
                // 验证事件
                if (!this._validateEmergencyEvent(event)) {
                    throw new Error('应急事件参数验证失败');
                }

                // 评估事件等级
                const severity = this._assessEventSeverity(event);
                
                // 启动响应
                const result = await this._initiateResponse({
                    ...event,
                    severity,
                    startTime: new Date()
                });
                
                // 记录事件
                await this._recordEmergencyEvent(result);

                // 如果是高级别事件，通知相关人员
                if (severity >= 2) {
                    await this._notifyStakeholders(result);
                }

                return result;
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'critical',
                    message: `启动应急响应失败: ${error.message}`,
                    data: event
                });
                throw error;
            }
        },

        // 更新应急响应状态
        updateEmergencyStatus: async function(eventId, status, resolution) {
            try {
                const event = await this._getEmergencyEvent(eventId);
                if (!event) {
                    throw new Error('应急事件不存在');
                }

                // 更新状态
                await this._updateEventStatus(eventId, status, resolution);
                
                // 记录更新
                await this._recordStatusUpdate(eventId, status, resolution);

                return { success: true, message: '应急响应状态已更新' };
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `更新应急响应状态失败: ${error.message}`,
                    eventId
                });
                throw error;
            }
        },

        // 内部方法：验证应急事件
        _validateEmergencyEvent: function(event) {
            return event && 
                   event.type && 
                   event.scope &&
                   event.level &&
                   event.description;
        },

        // 内部方法：评估事件等级
        _assessEventSeverity: function(event) {
            // 实现事件等级评估逻辑
            return BusinessProcessor.config.emergency.escalationLevels.indexOf(event.level);
        },

        // 内部方法：启动响应
        _initiateResponse: async function(event) {
            // 实现响应启动逻辑
            return {
                eventId: `EMG${Date.now()}`,
                status: 'responding',
                startTime: new Date()
            };
        },

        // 内部方法：通知相关人员
        _notifyStakeholders: async function(event) {
            // 实现通知逻辑
        },

        // 更新应急监控图表
        updateEmergencyMonitor: function(data) {
            const chart = document.querySelector('#emergency-monitor-chart');
            if (chart && chart.chart) {
                chart.chart.updateSeries([{
                    name: '响应时间',
                    data: data.responseTime
                }, {
                    name: '处理进度',
                    data: data.progress
                }]);
            }
        },

        // 更新应急事件记录
        updateEmergencyHistory: function(records) {
            const tableBody = document.querySelector('#emergency-history-table tbody');
            if (tableBody) {
                tableBody.innerHTML = records.map(record => `
                    <tr>
                        <td>${record.eventId}</td>
                        <td>${record.type}</td>
                        <td>${record.scope}</td>
                        <td><span class="badge bg-${
                            record.level === 'critical' ? 'danger' : 
                            record.level === 'high' ? 'warning' : 
                            record.level === 'medium' ? 'info' : 'success'
                        }">${record.level}</span></td>
                        <td>${record.startTime}</td>
                        <td><span class="badge bg-${record.status === 'resolved' ? 'success' : 'warning'}">${record.status}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary">详情</button>
                            ${record.status !== 'resolved' ? '<button class="btn btn-sm btn-warning">更新</button>' : ''}
                        </td>
                    </tr>
                `).join('');
            }
        }
    },

    // 告警处理器
    alertProcessor: {
        // 创建告警
        createAlert: function(alertData) {
            const alert = {
                id: this._generateAlertId(),
                timestamp: new Date(),
                level: alertData.level,
                message: alertData.message,
                data: alertData.data || {},
                status: 'active'
            };

            // 存储告警
            this._storeAlert(alert);

            // 如果是高级别告警，触发通知
            if (alert.level === 'error' || alert.level === 'critical') {
                this._notifyStakeholders(alert);
            }

            return alert;
        },

        // 更新告警状态
        updateAlertStatus: function(alertId, newStatus) {
            // 实现告警状态更新逻辑
        },

        // 获取活动告警
        getActiveAlerts: function() {
            // 实现获取活动告警逻辑
            return [];
        },

        // 内部方法：生成告警ID
        _generateAlertId: function() {
            return `ALT-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        },

        // 内部方法：存储告警
        _storeAlert: function(alert) {
            // 实现告警存储逻辑
        },

        // 内部方法：通知相关人员
        _notifyStakeholders: function(alert) {
            // 实现通知逻辑
        }
    },

    // 充电桩管理处理器
    chargingStationProcessor: {
        // 模拟数据
        _mockData: {
            chargers: [
                { id: 'CH001', type: 'DC', status: 'charging', power: 60, totalCharged: 1200, usageTime: 180, priority: 'high', health: 'good' },
                { id: 'CH002', type: 'DC', status: 'idle', power: 0, totalCharged: 800, usageTime: 120, priority: 'medium', health: 'good' },
                { id: 'CH003', type: 'AC', status: 'fault', power: 0, totalCharged: 500, usageTime: 90, priority: 'low', health: 'fault' }
            ],
            alerts: [
                { time: '2024-03-26 10:15:23', chargerId: 'CH003', type: 'hardware', level: 'critical', description: '充电模块故障', status: 'pending' },
                { time: '2024-03-26 11:30:45', chargerId: 'CH001', type: 'overload', level: 'warning', description: '负载接近上限', status: 'resolved' }
            ]
        },

        // 资源分配
        allocateResources: async function(stationId, strategy) {
            try {
                // 获取当前状态
                const status = await this._getStationStatus(stationId);
                
                // 根据策略生成分配方案
                const allocation = this._generateAllocation(status, strategy);
                
                // 执行分配
                await this._executeAllocation(allocation);
                
                // 更新显示
                this.updateStationStatus(stationId);
                
                return allocation;
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `资源分配失败: ${error.message}`,
                    data: { stationId, strategy }
                });
                throw error;
            }
        },

        // 更新优先级
        updatePriority: async function(chargerId, priority) {
            try {
                // 验证参数
                if (!this._validatePriorityUpdate(chargerId, priority)) {
                    throw new Error('优先级更新参数无效');
                }

                // 执行更新
                await this._executePriorityUpdate(chargerId, priority);
                
                // 更新显示
                this.updateChargerStatus(chargerId);

                return { success: true, message: '优先级更新成功' };
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `优先级更新失败: ${error.message}`,
                    data: { chargerId, priority }
                });
                throw error;
            }
        },

        // 处理异常
        handleException: async function(alert) {
            try {
                // 验证告警
                if (!this._validateAlert(alert)) {
                    throw new Error('告警数据无效');
                }

                // 生成处理方案
                const plan = this._generateHandlingPlan(alert);
                
                // 执行处理
                await this._executeHandlingPlan(plan);
                
                // 更新告警状态
                await this._updateAlertStatus(alert.id, 'handling');

                return plan;
            } catch (error) {
                BusinessProcessor.alertProcessor.createAlert({
                    level: 'error',
                    message: `异常处理失败: ${error.message}`,
                    data: alert
                });
                throw error;
            }
        },

        // 内部方法：获取充电站状态
        _getStationStatus: async function(stationId) {
            // 实现获取充电站状态的逻辑
            return {
                chargers: this._mockData.chargers,
                currentLoad: 180,
                maxLoad: 300,
                waitingVehicles: 3
            };
        },

        // 内部方法：生成分配方案
        _generateAllocation: function(status, strategy) {
            // 实现根据策略生成分配方案的逻辑
            return {
                assignments: [],
                expectedLoad: 0,
                estimatedWaitTime: 0
            };
        },

        // 内部方法：执行分配
        _executeAllocation: async function(allocation) {
            // 实现执行分配方案的逻辑
        },

        // 内部方法：验证优先级更新
        _validatePriorityUpdate: function(chargerId, priority) {
            return chargerId && priority && 
                   ['high', 'medium', 'low'].includes(priority);
        },

        // 内部方法：执行优先级更新
        _executePriorityUpdate: async function(chargerId, priority) {
            // 实现优先级更新的逻辑
        },

        // 内部方法：验证告警
        _validateAlert: function(alert) {
            return alert && alert.type && alert.level && alert.description;
        },

        // 内部方法：生成处理方案
        _generateHandlingPlan: function(alert) {
            // 实现生成处理方案的逻辑
            return {
                steps: [],
                estimatedTime: 0,
                requiredResources: []
            };
        },

        // 内部方法：执行处理方案
        _executeHandlingPlan: async function(plan) {
            // 实现执行处理方案的逻辑
        },

        // 更新充电站状态显示
        updateStationStatus: function(stationId) {
            // 更新状态指标
            document.getElementById('current-usage').textContent = '75%';
            document.getElementById('waiting-vehicles').textContent = '3辆';
            document.getElementById('fault-chargers').textContent = '1个';
            document.getElementById('avg-wait-time').textContent = '8分钟';

            // 更新充电桩状态表格
            const tableBody = document.querySelector('#charger-status-table tbody');
            if (tableBody) {
                tableBody.innerHTML = this._mockData.chargers.map(charger => `
                    <tr>
                        <td>${charger.id}</td>
                        <td>${charger.type}</td>
                        <td><span class="badge bg-${
                            charger.status === 'charging' ? 'success' : 
                            charger.status === 'idle' ? 'info' : 'danger'
                        }">${charger.status}</span></td>
                        <td>${charger.power} kW</td>
                        <td>${charger.totalCharged} kWh</td>
                        <td>${charger.usageTime} min</td>
                        <td><span class="badge bg-${
                            charger.priority === 'high' ? 'danger' : 
                            charger.priority === 'medium' ? 'warning' : 'info'
                        }">${charger.priority}</span></td>
                        <td><span class="badge bg-${
                            charger.health === 'good' ? 'success' : 'danger'
                        }">${charger.health}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary view-charger" data-id="${charger.id}">详情</button>
                            ${charger.status !== 'fault' ? 
                                `<button class="btn btn-sm btn-warning ms-1 adjust-priority" data-id="${charger.id}">调整优先级</button>` : 
                                `<button class="btn btn-sm btn-danger ms-1 handle-fault" data-id="${charger.id}">处理故障</button>`
                            }
                        </td>
                    </tr>
                `).join('');
            }

            // 更新告警表格
            const alertTableBody = document.querySelector('#charging-alert-table tbody');
            if (alertTableBody) {
                alertTableBody.innerHTML = this._mockData.alerts.map(alert => `
                    <tr>
                        <td>${alert.time}</td>
                        <td>${alert.chargerId}</td>
                        <td>${alert.type}</td>
                        <td><span class="badge bg-${
                            alert.level === 'critical' ? 'danger' : 
                            alert.level === 'warning' ? 'warning' : 'info'
                        }">${alert.level}</span></td>
                        <td>${alert.description}</td>
                        <td><span class="badge bg-${
                            alert.status === 'resolved' ? 'success' : 
                            alert.status === 'handling' ? 'warning' : 'danger'
                        }">${alert.status}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary view-alert" data-id="${alert.chargerId}">详情</button>
                            ${alert.status === 'pending' ? 
                                `<button class="btn btn-sm btn-danger ms-1 handle-alert" data-id="${alert.chargerId}">处理</button>` : 
                                ''
                            }
                        </td>
                    </tr>
                `).join('');
            }
        },

        // 初始化充电负荷趋势图
        initChargingLoadChart: function() {
            return new ApexCharts(document.querySelector("#charging-load-chart"), {
                series: [{
                    name: '实际负荷',
                    data: []
                }, {
                    name: '预测负荷',
                    data: []
                }],
                chart: {
                    type: 'line',
                    height: 300
                },
                xaxis: {
                    type: 'datetime'
                },
                yaxis: {
                    title: {
                        text: '负荷 (kW)'
                    }
                }
            });
        }
    },

    // 记录处理器
    recordProcessor: {
        // 模拟数据
        _mockData: {
            dispatch: [
                { id: 'DSP20240326001', type: '发电调度', target: '风机组 #1', value: 50, executeTime: '2024-03-22 10:00', responseTime: '2.5s', executionRate: '98.5%', status: 'completed' },
                { id: 'DSP20240326002', type: '启停控制', target: '光伏组#B08', value: '停机', executeTime: '2024-03-26 11:30:45', responseTime: '3.1s', executionRate: '97.8%', status: 'executing' },
                { id: 'DSP20240326003', type: '功率调节', target: '储能站#C03', value: '200MW→300MW', executeTime: '2024-03-26 13:20:18', responseTime: '2.8s', executionRate: '99.1%', status: 'completed' }
            ],
            trading: [
                { id: 'TRD20240326001', market: '日前市场', type: '现货交易', direction: '售电', price: 485.5, volume: 1000, amount: 485500, executeTime: '2024-03-26 09:00:00', status: 'completed' },
                { id: 'TRD20240326002', market: '实时市场', type: '实时交易', direction: '购电', price: 520.8, volume: 500, amount: 260400, executeTime: '2024-03-26 10:30:00', status: 'executing' },
                { id: 'TRD20240326003', market: '辅助服务', type: '调频服务', direction: '售出', price: 600.0, volume: 200, amount: 120000, executeTime: '2024-03-26 14:15:00', status: 'pending' }
            ],
            maintenance: [
                { id: 'MNT20240326001', device: '变压器#T12', type: '定期检修', level: '二级', planTime: '2024-03-26 08:00:00', actualTime: '2024-03-26 08:15:00', duration: '4h', status: 'completed' },
                { id: 'MNT20240326002', device: '断路器#C05', type: '预防性检修', level: '一级', planTime: '2024-03-26 13:00:00', actualTime: '2024-03-26 13:10:00', duration: '2h', status: 'executing' },
                { id: 'MNT20240326003', device: '母线#B02', type: '临时检修', level: '三级', planTime: '2024-03-26 15:00:00', actualTime: null, duration: '3h', status: 'pending' }
            ],
            charging: [
                { id: 'CHG20240326001', chargerId: 'CH001', user: '用户A', power: 45.6, startTime: '2024-03-26 09:15:23', endTime: '2024-03-26 10:45:23', cost: 152.5, status: 'completed' },
                { id: 'CHG20240326002', chargerId: 'CH003', user: '用户B', power: 30.2, startTime: '2024-03-26 11:20:45', endTime: null, cost: null, status: 'charging' },
                { id: 'CHG20240326003', chargerId: 'CH002', user: '用户C', power: 25.8, startTime: '2024-03-26 13:10:18', endTime: '2024-03-26 14:25:18', cost: 86.3, status: 'completed' }
            ]
        },

        // 显示记录
        displayRecords: function(type) {
            const records = this._mockData[type] || [];
            const tableBody = document.getElementById(`${type}-records-body`);
            if (!tableBody) return;

            let html = '';
            records.forEach(record => {
                html += this._generateRecordRow(type, record);
            });
            tableBody.innerHTML = html;
        },

        // 生成记录行
        _generateRecordRow: function(type, record) {
            const statusClass = this._getStatusClass(record.status);
            let columns = '';

            switch(type) {
                case 'dispatch':
                    columns = `
                        <td>${record.id}</td>
                        <td>${record.type}</td>
                        <td>${record.target}</td>
                        <td>${record.value}</td>
                        <td>${record.executeTime}</td>
                        <td>${record.responseTime}</td>
                        <td>${record.executionRate}</td>
                        <td><span class="badge bg-${statusClass}">${record.status}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary view-record" data-id="${record.id}">查看</button>
                            ${record.status !== 'completed' ? `<button class="btn btn-sm btn-danger ms-1 cancel-record" data-id="${record.id}">取消</button>` : ''}
                        </td>
                    `;
                    break;
                case 'trading':
                    columns = `
                        <td>${record.id}</td>
                        <td>${record.market}</td>
                        <td>${record.type}</td>
                        <td>${record.direction}</td>
                        <td>${record.price}</td>
                        <td>${record.volume}</td>
                        <td>${record.amount}</td>
                        <td>${record.executeTime}</td>
                        <td><span class="badge bg-${statusClass}">${record.status}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary view-record" data-id="${record.id}">查看</button>
                            ${record.status !== 'completed' ? `<button class="btn btn-sm btn-danger ms-1 cancel-record" data-id="${record.id}">取消</button>` : ''}
                        </td>
                    `;
                    break;
                case 'maintenance':
                    columns = `
                        <td>${record.id}</td>
                        <td>${record.device}</td>
                        <td>${record.type}</td>
                        <td>${record.level}</td>
                        <td>${record.planTime}</td>
                        <td>${record.actualTime || '-'}</td>
                        <td>${record.duration}</td>
                        <td><span class="badge bg-${statusClass}">${record.status}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary view-record" data-id="${record.id}">查看</button>
                            ${record.status !== 'completed' ? `<button class="btn btn-sm btn-danger ms-1 cancel-record" data-id="${record.id}">取消</button>` : ''}
                        </td>
                    `;
                    break;
                case 'charging':
                    columns = `
                        <td>${record.id}</td>
                        <td>${record.chargerId}</td>
                        <td>${record.user}</td>
                        <td>${record.power}</td>
                        <td>${record.startTime}</td>
                        <td>${record.endTime || '-'}</td>
                        <td>${record.cost ? `¥${record.cost}` : '-'}</td>
                        <td><span class="badge bg-${statusClass}">${record.status}</span></td>
                        <td>
                            <button class="btn btn-sm btn-primary view-record" data-id="${record.id}">查看</button>
                            ${record.status !== 'completed' ? `<button class="btn btn-sm btn-danger ms-1 cancel-record" data-id="${record.id}">取消</button>` : ''}
                        </td>
                    `;
                    break;
            }

            return `<tr>${columns}</tr>`;
        },

        // 获取状态样式类
        _getStatusClass: function(status) {
            switch(status) {
                case 'completed': return 'success';
                case 'executing': 
                case 'charging': return 'warning';
                case 'pending': return 'info';
                default: return 'secondary';
            }
        }
    },

    // 页面初始化
    init: function() {
        console.log('初始化业务处理模块...');
        
        // 显示初始记录（调度记录）
        this.recordProcessor.displayRecords('dispatch');
        
        // 绑定标签页切换事件
        document.querySelectorAll('a[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const type = e.target.getAttribute('href').replace('#', '').split('-')[0];
                if (['dispatch', 'trading', 'maintenance', 'charging'].includes(type)) {
                    this.recordProcessor.displayRecords(type);
                }
            });
        });
    }
};

// Export the necessary functions and objects
window.BusinessProcessor = {
    ...BusinessProcessor,
    
    // Add modal integration functions
    modalIntegration: {
        // Initialize modal-related functionality
    init: function() {
            try {
                // Bind view button events
                document.querySelectorAll('.view-record-btn').forEach(btn => {
                    btn.addEventListener('click', function(e) {
                        const recordId = this.getAttribute('data-record-id');
                        console.log('View button clicked for record:', recordId);
                        // Additional view logic here
                    });
                });

                // Initialize filter functionality
                initializeFilter();
                
                // Bind filter button events
                const applyFilterBtn = document.getElementById('apply-filter');
                const resetFilterBtn = document.getElementById('reset-filter');
                
                if (applyFilterBtn) {
                    applyFilterBtn.addEventListener('click', this.applyFilter.bind(this));
                }
                if (resetFilterBtn) {
                    resetFilterBtn.addEventListener('click', this.resetFilter.bind(this));
                }
                
                console.log('Modal integration initialized successfully');
            } catch (error) {
                console.error('Error initializing modal integration:', error);
            }
        },

        // Re-initialize events after dynamic content updates
        refreshEvents: function() {
            this.init();
            console.log('Modal events refreshed');
        },

        // Initialize filter functionality (moved from global scope)
        initializeFilter: initializeFilter,
        
        // Filter application function (moved from global scope)
        applyFilter: applyFilter,
        
        // Reset filter function (moved from global scope)
        resetFilter: resetFilter
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    try {
        if (window.BusinessProcessor && window.BusinessProcessor.modalIntegration) {
            window.BusinessProcessor.modalIntegration.init();
            console.log('BusinessProcessor initialized');
            } else {
            console.warn('BusinessProcessor or modalIntegration not available');
        }
    } catch (error) {
        console.error('Error during BusinessProcessor initialization:', error);
    }
});