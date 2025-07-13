    const strategyManager = {
    // All strategy data, now directly inside the manager
        strategies: {
            renewable: [
                {
                    id: 1,
                    name: "风电短期功率预测",
                type: "wind", // Keep specific type for internal logic if needed
                displayType: "风电", // For display
                    timeScale: "短期(0-4h)",
                    updateTime: "2024-03-21 15:30",
                    status: "enabled",
                description: "基于高频气象数据和SCADA数据的风电功率预测，优化实时调度。",
                accuracy: "92.5%", // More specific
                    algorithms: ["LSTM", "Transformer"],
                dataSource: ["高频气象数据 (风速、风向、温度)", "SCADA历史功率", "机组状态"],
                parameters: {
                    predictionHorizon: 4, // hours
                    updateInterval: 0.25, // hours (15 min)
                    confidenceLevel: 0.95,
                    rampRateLimit: 0.1, // MW/min
                    curtailmentThreshold: 0.9, // % of capacity
                    smoothingFactor: 0.2 // For output smoothing
                },
                constraints: {
                    gridCodeCompliance: ["频率响应", "电压支撑"],
                    environmentalRestrictions: ["噪音限制时段"]
                },
                objectives: ["minimizeMAPE", "maximizeGridStabilitySupport", "predictRampEvents"],
                version: "v1.1.0",
                author: "系统预置",
                lastRunTime: "2024-03-22 10:00",
                runCount: 150,
                performanceMetrics: { // Detailed metrics
                    mape: 7.5, // Mean Absolute Percentage Error
                    rmse: 1.2, // Root Mean Square Error (MW)
                    rampPredictionAccuracy: 85 // %
                }
                },
                {
                    id: 2,
                    name: "光伏日前预测",
                    type: "solar",
                displayType: "光伏",
                    timeScale: "日前(24h)",
                    updateTime: "2024-03-21 14:20",
                    status: "enabled",
                description: "基于NWP气象预报和卫星云图的光伏集群发电量预测，用于日前计划。",
                accuracy: "89.0%",
                algorithms: ["CNN", "Prophet", "LightGBM"],
                dataSource: ["NWP气象数据 (辐照度GHI/DNI/DHI, 温度, 云量)", "卫星云图", "历史发电量", "逆变器效率曲线"],
                parameters: {
                    predictionHorizon: 24, // hours
                    updateInterval: 6, // hours
                    confidenceLevel: 0.90,
                    soilingLossFactor: 0.02, // % loss due to soiling
                    temperatureDerating: true // Consider temperature effects
                },
                constraints: {
                    gridCodeCompliance: ["功率因数要求"],
                    inverterCapacityLimit: 50 // MW
                },
                objectives: ["minimizeRMSE", "accuratePeakPrediction", "informDayAheadScheduling"],
                version: "v1.0.5",
                author: "系统预置",
                lastRunTime: "2024-03-22 08:00",
                runCount: 120,
                 performanceMetrics: {
                    mape: 11.0,
                    rmse: 3.5, // MW
                    peakAccuracy: 92 // %
                }
                },
                {
                    id: 3,
                name: "风光联合优化调度", // Changed name slightly
                    type: "hybrid",
                displayType: "风光联合",
                    timeScale: "超短期(15min)",
                    updateTime: "2024-03-21 13:15",
                    status: "enabled",
                description: "风光储多能源协同预测与优化调度，提升整体可预测性和经济性。",
                accuracy: "94.2%", // Overall system predictability
                algorithms: ["Transformer", "XGBoost", "MPC (Model Predictive Control)"],
                dataSource: ["风电预测结果", "光伏预测结果", "储能状态(SOC)", "电价信号", "电网调度指令"],
                parameters: {
                    predictionHorizon: 0.25, // hours
                    controlInterval: 0.08, // hours (5 min)
                    confidenceLevel: 0.98,
                    storageEfficiency: 0.9, // Round-trip efficiency
                    socLimits: { min: 0.1, max: 0.9 } // State of Charge limits
                },
                constraints: {
                    pccPowerLimit: 100, // MW limit at Point of Common Coupling
                    storageChargeDischargeRate: 20 // MW
                },
                objectives: ["maximizeRevenue", "minimizeDeviationPenalty", "smoothOutputPower"],
                version: "v2.0.1",
                author: "系统预置",
                lastRunTime: "2024-03-22 11:15",
                runCount: 300,
                performanceMetrics: {
                    deviationRMSE: 0.8, // MW
                    revenueIncrease: 8, // % vs independent operation
                    curtailmentReduction: 15 // %
                }
                }
            ],
            load: [
                {
                    id: 1,
                name: "工业负荷短期预测",
                    type: "industrial",
                displayType: "工业",
                predictionCycle: "日前", // Can be multi-scale (e.g., "日前+日内")
                factors: ["历史负荷", "温度", "湿度", "工作日/周末", "特殊排班", "电价"], // More factors
                features: ["lagged_load_1h", "lagged_load_24h", "temp_forecast", "humidity_forecast", "day_of_week", "hour_of_day", "is_holiday", "price_signal"], // Example features
                    updateTime: "2024-03-21 15:30",
                    status: "enabled",
                description: "预测大型工业园区或特定工厂的用电负荷，考虑生产计划影响。",
                accuracy: "91.3%",
                algorithms: ["LSTM", "SVR", "Attention-Based CNN"],
                parameters: {
                    seasonality: 'weekly, daily',
                    trend: 'piecewise_linear',
                    weatherSensitivity: 0.7, // Factor for temperature impact
                    holidayEffect: true, // Enable holiday modeling
                    specialEvents: ["maintenance_shutdown", "production_rampup"] // List of event types
                },
                constraints: {
                    maxLoadLimit: 150 // MW
                },
                objectives: ["minimizeMAPE", "accuratePeakLoadTiming", "reduceDemandCharges"],
                version: "v1.2.0",
                author: "系统预置",
                lastRunTime: "2024-03-22 09:00",
                runCount: 180,
                performanceMetrics: {
                    mape: 8.7,
                    peakTimeError: 0.5 // hours
                }
                },
                {
                    id: 2,
                name: "商业区域负荷预测",
                    type: "commercial",
                displayType: "商业",
                predictionCycle: "周前+日前",
                factors: ["历史负荷", "温度", "湿度", "节假日", "商场活动", "人流量"],
                features: ["lagged_load_24h", "lagged_load_168h", "temp_forecast", "humidity_forecast", "holiday_flag", "event_flag", "mobility_index"],
                    updateTime: "2024-03-21 14:20",
                    status: "enabled",
                description: "预测购物中心、写字楼等商业区域的用电负荷，考虑人群活动。",
                accuracy: "88.5%",
                algorithms: ["Prophet", "XGBoost", "SARIMA"],
                parameters: {
                    holidayEffect: true,
                    changepointPriorScale: 0.05,
                    weatherSensitivity: 0.6,
                    mobilityDataIntegration: true // Use mobility data if available
                },
                constraints: {
                    minLoadRequirement: 5 // MW (e.g., for essential services)
                },
                objectives: ["minimizeRMSE", "predictWeekendPeak", "supportHVACoptimization"],
                version: "v1.0.0",
                author: "系统预置",
                lastRunTime: "2024-03-18 10:00",
                runCount: 50,
                 performanceMetrics: {
                    mape: 11.5,
                    rmse: 2.1 // MW
                }
                },
                {
                    id: 3,
                name: "居民负荷日内预测",
                    type: "residential",
                displayType: "居民",
                    predictionCycle: "日内",
                factors: ["历史负荷", "温度", "湿度", "时间", "辐照度(影响空调)"],
                features: ["lagged_load_15min", "lagged_load_1h", "temp_forecast_15min", "humidity_forecast_15min", "solar_irradiance_forecast", "hour_of_day", "minute_of_hour"],
                    updateTime: "2024-03-21 13:15",
                    status: "enabled",
                description: "高频预测居民区用电负荷，用于实时平衡和电压管理。",
                accuracy: "90.1%",
                algorithms: ["GRU", "Random Forest", "LightGBM"],
                parameters: {
                    timeOfDayFeatures: true,
                    temperatureLag: 1, // hours
                    weatherSensitivity: 0.8
                },
                constraints: {
                     dataLatencyTolerance: 5 // minutes
                },
                objectives: ["minimizeMAE", "realtimeBalancingSupport", "voltageProfilePrediction"],
                version: "v1.1.0",
                author: "系统预置",
                lastRunTime: "2024-03-22 11:00",
                runCount: 250,
                 performanceMetrics: {
                    mae: 0.5, // Mean Absolute Error (MW)
                    shortTermMAPE: 9.9 // MAPE for next 15-60 min
                }
                }
            ],
            charging: [
                {
                    id: 1,
                name: "公交场站充电负荷预测",
                    type: "bus",
                displayType: "公交枢纽",
                predictionRange: "日前24h+日内4h",
                factors: ["历史充电数据", "公交运营计划", "车辆SOC", "电价", "天气"],
                features: ["vehicle_schedule", "initial_soc", "arrival_time_distribution", "energy_consumption_per_route", "time_of_day_price", "temperature"],
                    updateTime: "2024-03-21 15:30",
                    status: "enabled",
                description: "预测公交车队的充电需求和负荷曲线，支持有序充电调度。",
                accuracy: "93.4%", // Charging demand accuracy
                algorithms: ["LSTM", "XGBoost", "Queueing Theory Model"],
                parameters: {
                    vehicleCount: 50,
                    routeComplexity: 'medium',
                    chargingPowerLevels: [60, 120], // kW
                    scheduleAdherenceRate: 0.95,
                    priceElasticity: -0.1 // How demand changes with price
                },
                constraints: {
                     stationCapacity: 1.5, // MW
                     chargerAvailability: 0.98, // % of chargers working
                     maxChargeTimePerVehicle: 4 // hours
                },
                objectives: ["minimizeChargingCost", "meetScheduleRequirements", "predictPeakChargingLoad"],
                version: "v1.0.0",
                author: "系统预置",
                lastRunTime: "2024-03-22 07:30",
                runCount: 100,
                performanceMetrics: {
                    loadMAPE: 6.6, // Load prediction MAPE
                    costReduction: 12 // % compared to uncontrolled charging
                }
                },
                {
                    id: 2,
                name: "公共充电站负荷预测",
                    type: "public",
                displayType: "公共充电站",
                predictionRange: "日内4h+日前24h",
                factors: ["历史充电数据", "时间", "天气", "周边POI", "交通流量", "电价"],
                features: ["lagged_station_load", "hour_of_day", "day_of_week", "is_holiday", "temperature", "precipitation", "nearby_event_flag", "traffic_congestion_index", "realtime_price"],
                    updateTime: "2024-03-21 14:20",
                    status: "enabled",
                description: "预测公共充电站（快充、慢充）的充电负荷和排队情况。",
                accuracy: "87.8%",
                algorithms: ["Prophet", "Random Forest", "Spatial-Temporal GNN"],
                parameters: {
                    locationType: 'commercial', // or 'highway', 'residential_area'
                    weatherSensitivity: 0.8,
                    poiInfluenceRadius: 500, // meters
                    userBehaviorModel: 'price_sensitive_commuter' // Example model
                },
                constraints: {
                    gridCapacityLimit: 1, // MW at connection point
                    transformerLoadLimit: 0.8, // % of transformer capacity
                },
                objectives: ["maximizeStationUtilization", "minimizeWaitingTimes", "provideGridSupportSignal"],
                version: "v1.1.2",
                author: "系统预置",
                lastRunTime: "2024-03-22 10:30",
                runCount: 200,
                performanceMetrics: {
                    utilizationRate: 65, // %
                    averageWaitingTime: 10 // minutes
                }
                },
                {
                    id: 3,
                name: "私人充电桩聚合预测",
                    type: "private",
                displayType: "私人充电桩",
                predictionRange: "周前+日前",
                factors: ["历史充电数据", "用户习惯画像", "电价政策", "车辆类型"],
                features: ["user_profile_segment", "time_since_last_charge", "typical_charge_duration", "soc_at_plugin_distribution", "tou_price_schedule", "vehicle_battery_capacity"],
                    updateTime: "2024-03-21 13:15",
                    status: "enabled",
                description: "预测大量私人充电桩的聚合充电行为和负荷。",
                accuracy: "89.5%", // Aggregate load accuracy
                algorithms: ["GRU", "SVR", "Agent-Based Modeling"],
                parameters: {
                    userGroup: 'commuter',
                    weekendMultiplier: 1.5,
                    smartChargingParticipationRate: 0.6, // % of users allowing smart charging
                    defaultChargingStartTime: "19:00"
                },
                constraints: {
                    distributionNetworkLimits: ["voltage_drop", "transformer_aging"]
                },
                objectives: ["predictAggregateLoadProfile", "estimateFlexibilityPotential", "supportDemandResponse"],
                version: "v1.0.1",
                author: "系统预置",
                lastRunTime: "2024-03-19 12:00",
                runCount: 40,
                 performanceMetrics: {
                    aggregateLoadRMSE: 150, // kW (for a given region)
                    flexibilityOffered: 200 // kW (potential load shift)
                }
                }
            ],
            trading: [
                {
                    id: 1,
                name: "峰谷分时电价套利",
                    type: "price",
                displayType: "分时电价",
                profitRate: "15.2%", // Example profit rate
                    riskLevel: "low",
                    updateTime: "2024-03-21 15:30",
                    status: "enabled",
                description: "基于固定或动态时段电价差，通过储能充放电进行低风险套利。",
                parameters: {
                    minAmount: 1, // MWh per cycle
                    maxAmount: 10, // MWh per cycle
                    timeSlots: [ // Example detailed slots
                        {type: 'valley', startTime: '00:00', endTime: '07:00', price: 0.35},
                        {type: 'normal', startTime: '07:00', endTime: '10:00', price: 0.80},
                        {type: 'peak', startTime: '10:00', endTime: '12:00', price: 1.25},
                        {type: 'normal', startTime: '12:00', endTime: '18:00', price: 0.85},
                        {type: 'peak', startTime: '18:00', endTime: '21:00', price: 1.30},
                        {type: 'normal', startTime: '21:00', endTime: '24:00', price: 0.80}
                    ],
                    minPriceDifferenceThreshold: 0.5, // Minimum price diff (peak-valley) to trigger trade
                    storageEfficiency: 0.88 // Round-trip efficiency assumption
                },
                constraints: {
                    storageCycleLifeDegradation: 0.0001, // % degradation per cycle
                    maxDailyCycles: 2
                },
                objectives: ["maximizeArbitrageProfit", "minimizeStorageDegradation", "operateWithinSOCLimits"],
                version: "v1.3.0",
                author: "系统预置",
                lastRunTime: "2024-03-22 09:30",
                runCount: 90,
                successRate: "95.5%", // % of profitable cycles
                performanceMetrics: {
                    averageProfitPerCycle: 500, // Currency unit
                    annualizedROI: 15.2 // %
                }
                },
                {
                    id: 2,
                name: "电力现货市场日前申报",
                    type: "spot",
                displayType: "现货市场",
                profitRate: "22.1%",
                    riskLevel: "medium",
                    updateTime: "2024-03-21 14:20",
                    status: "enabled",
                description: "基于日前市场价格预测和自身成本/发电预测进行报价申报。",
                parameters: {
                    minAmount: 5, // MWh per hour block
                    maxAmount: 50, // MWh per hour block
                    priceVolatilityThreshold: 0.1, // Price forecast uncertainty threshold
                    marketDepthConsideration: true, // Adjust bidding based on expected market volume
                    generationCostModel: "linear", // or 'quadratic'
                    riskAversionLevel: 0.3 // 0=neutral, 1=max aversion
                },
                 constraints: {
                     positionLimits: 100, // MWh maximum net position
                     regulatoryCompliance: ["market_rules_v3.2"],
                     creditLimit: 1000000 // Currency unit
                 },
                 objectives: ["maximizeProfit", "minimizeRisk (VaR_95)", "maintainMarketCompliance"],
                version: "v2.1.0",
                author: "系统预置",
                lastRunTime: "2024-03-22 11:00",
                runCount: 150,
                successRate: "85.3%", // % of blocks cleared profitably vs forecast
                 performanceMetrics: {
                     averageClearedPricePremium: 0.05, // $/MWh vs average market price
                     valueAtRisk_95: 50000 // 95% VaR in currency units
                 }
                },
                {
                    id: 3,
                name: "调频辅助服务响应",
                    type: "auxiliary",
                displayType: "辅助服务(调频)",
                profitRate: "18.0%",
                riskLevel: "high", // Can be high due to penalties
                    updateTime: "2024-03-21 13:15",
                    status: "enabled",
                description: "利用储能或可调负荷快速响应电网频率偏差，提供调频辅助服务。",
                parameters: {
                    minAmount: 2, // MW capacity offered
                    maxAmount: 20, // MW capacity offered
                    responseTime: 1, // Required response time in seconds
                    frequencyDeviationTolerance: 0.05, // Hz deadband
                    serviceDuration: 15, // minutes per event typically
                    stateOfChargeReservation: 0.2 // Reserve SOC for service
                },
                constraints: {
                    performanceAccuracyRequirement: 95, // % accuracy required by grid operator
                    communicationLatency: 0.5 // seconds max allowed latency
                },
                objectives: ["maximizeServiceRevenue", "meetPerformanceStandards", "minimizeSOCDeviation"],
                version: "v1.0.2",
                author: "系统预置",
                lastRunTime: "2024-03-22 06:00",
                runCount: 60,
                successRate: "90.0%", // % of service calls successfully met performance requirements
                performanceMetrics: {
                    averageResponseTime: 0.8, // seconds
                    performanceScore: 96.5 // % based on grid operator assessment
                }
            }
        ]
    },

    // Get next available ID for a strategy type
    _getNextId(type) {
        const strategiesOfType = this.strategies[type] || [];
        if (strategiesOfType.length === 0) {
            return 1;
        }
        return Math.max(...strategiesOfType.map(s => s.id)) + 1;
    },

    // Centralized function to create a strategy (dispatcher)
    createStrategy(type) {
        console.log(`Creating new strategy of type: ${type}`);
        switch (type) {
            case 'renewable':
                this._showCreateEditModal(type);
                break;
            case 'load':
                this._showCreateEditModal(type);
                break;
            case 'charging':
                this._showCreateEditModal(type);
                break;
            case 'trading':
                this._showCreateTradingModal(); // Use specialized modal for trading for now
                break;
            default:
                console.error(`Unknown strategy type: ${type}`);
                Swal.fire('错误', '未知的策略类型', 'error');
        }
    },

    // Generic create/edit modal (basic version)
    _showCreateEditModal(type, strategyId = null) {
        const isEditing = strategyId !== null;
        const strategy = isEditing ? this.strategies[type].find(s => s.id === strategyId) : null;
        const title = isEditing ? `编辑 ${this._getTypeDisplayName(type)} 策略` : `新建 ${this._getTypeDisplayName(type)} 策略`;

        // --- Basic Fields ---
        let htmlContent = `
            <form id="strategyForm" class="text-start needs-validation" novalidate>
                        <div class="mb-3">
                            <label class="form-label">策略名称</label>
                    <input type="text" class="form-control" id="strategyName" value="${strategy?.name || ''}" required>
                    <div class="invalid-feedback">请输入策略名称</div>
                        </div>
                        <div class="mb-3">
                    <label class="form-label">策略描述</label>
                    <textarea class="form-control" id="strategyDescription" rows="3" required>${strategy?.description || ''}</textarea>
                     <div class="invalid-feedback">请输入策略描述</div>
                </div>
                `;

        // --- Type-Specific Fields (Add more detail later) ---
        if (type === 'renewable') {
            htmlContent += `
                <div class="mb-3">
                    <label class="form-label">时间尺度</label>
                    <select class="form-select" id="strategyTimeScale">
                        <option value="超短期(15min)" ${strategy?.timeScale === '超短期(15min)' ? 'selected' : ''}>超短期(15min)</option>
                        <option value="短期(0-4h)" ${strategy?.timeScale === '短期(0-4h)' ? 'selected' : ''}>短期(0-4h)</option>
                        <option value="日前(24h)" ${strategy?.timeScale === '日前(24h)' ? 'selected' : ''}>日前(24h)</option>
                            </select>
                        </div>
                `;
        } else if (type === 'load') {
             htmlContent += `
                        <div class="mb-3">
                    <label class="form-label">预测周期</label>
                     <select class="form-select" id="strategyPredictionCycle">
                        <option value="日内" ${strategy?.predictionCycle === '日内' ? 'selected' : ''}>日内</option>
                        <option value="日前" ${strategy?.predictionCycle === '日前' ? 'selected' : ''}>日前</option>
                        <option value="周前" ${strategy?.predictionCycle === '周前' ? 'selected' : ''}>周前</option>
                    </select>
                        </div>
                        <div class="mb-3">
                     <label class="form-label">影响因素 (暂不可编辑)</label>
                     <div class="d-flex flex-wrap gap-1">
                         ${(strategy?.factors || ['历史负荷', '温度']).map(factor => `<span class="badge bg-info">${factor}</span>`).join('')}
                            </div>
                            </div>
                `;
        } else if (type === 'charging') {
             htmlContent += `
                        <div class="mb-3">
                    <label class="form-label">预测范围</label>
                     <select class="form-select" id="strategyPredictionRange">
                        <option value="日内4h" ${strategy?.predictionRange === '日内4h' ? 'selected' : ''}>日内4h</option>
                        <option value="日前24h" ${strategy?.predictionRange === '日前24h' ? 'selected' : ''}>日前24h</option>
                        <option value="周前" ${strategy?.predictionRange === '周前' ? 'selected' : ''}>周前</option>
                    </select>
                            </div>
                `;
        }
        // Add more specific fields for algorithms, data sources, parameters later

        htmlContent += `</form>`;

        Swal.fire({
            title: title,
            html: htmlContent,
                showCancelButton: true,
            confirmButtonText: isEditing ? '保存' : '创建',
                cancelButtonText: '取消',
            didOpen: () => {
                // Initialize validation if needed
            },
                preConfirm: () => {
                const form = document.getElementById('strategyForm');
                    if (!form.checkValidity()) {
                    // Trigger Bootstrap validation UI
                    form.classList.add('was-validated');
                    // Prevent Swal from closing
                        return false;
                    }

                const name = document.getElementById('strategyName').value;
                const description = document.getElementById('strategyDescription').value;
                let specificData = {};

                if (type === 'renewable') {
                    specificData.timeScale = document.getElementById('strategyTimeScale').value;
                } else if (type === 'load') {
                    specificData.predictionCycle = document.getElementById('strategyPredictionCycle').value;
                    // Factors are not editable in this basic version
                } else if (type === 'charging') {
                     specificData.predictionRange = document.getElementById('strategyPredictionRange').value;
                }


                return { name, description, ...specificData };
                }
            }).then((result) => {
                if (result.isConfirmed) {
                const data = result.value;
                if (isEditing) {
                    // Find and update the strategy
                    const index = this.strategies[type].findIndex(s => s.id === strategyId);
                    if (index !== -1) {
                        this.strategies[type][index] = {
                            ...this.strategies[type][index], // Keep existing properties
                            name: data.name,
                            description: data.description,
                            ...(type === 'renewable' && { timeScale: data.timeScale }),
                            ...(type === 'load' && { predictionCycle: data.predictionCycle }),
                            ...(type === 'charging' && { predictionRange: data.predictionRange }),
                            updateTime: new Date().toLocaleString('sv-SE') // Use a sortable format
                        };
                         Swal.fire('保存成功', '策略信息已更新', 'success');
                    } else {
                         Swal.fire('错误', '未找到要编辑的策略', 'error');
                         return;
                    }
                } else {
                    // Create new strategy
                     const newId = this._getNextId(type);
                    const newStrategy = {
                        id: newId,
                        name: data.name,
                        description: data.description,
                        type: 'custom', // Mark as custom
                        status: 'enabled', // Default status
                        updateTime: new Date().toLocaleString('sv-SE'),
                        version: 'v1.0.0',
                        author: '当前用户', // Replace with actual user later
                        accuracy: 'N/A',
                        lastRunTime: 'N/A',
                        runCount: 0,
                        algorithms: ['Default'], // Placeholder
                        dataSource: ['Default'], // Placeholder
                        parameters: {}, // Placeholder
                        ...(type === 'renewable' && { timeScale: data.timeScale, displayType: '自定义' }),
                        ...(type === 'load' && { predictionCycle: data.predictionCycle, displayType: '自定义', factors: ['历史负荷', '温度'] }), // Default factors
                        ...(type === 'charging' && { predictionRange: data.predictionRange, displayType: '自定义', factors: ['历史充电', '时间'] }), // Default factors
                    };
                    this.strategies[type].push(newStrategy);
                    Swal.fire('创建成功', '新策略已创建', 'success');
                }
                this.refreshTables(); // Refresh tables after create/edit
                }
            });
        },

    // Specific modal for creating/editing Trading strategies (retains previous detail)
    _showCreateTradingModal(strategyId = null) {
        const isEditing = strategyId !== null;
        const strategy = isEditing ? this.strategies.trading.find(s => s.id === strategyId) : null;
        const title = isEditing ? '编辑能源交易策略' : '新建能源交易策略';

        // Helper function to generate time slot HTML
        const generateTimeSlotHtml = (slotData = { type: 'normal', startTime: '', endTime: '', price: '' }) => `
            <div class="price-slot mb-2">
                <div class="d-flex gap-2 align-items-center">
                    <select class="form-select form-select-sm w-25" required ${isEditing ? '' : ''}> <!-- Name attribute removed, validation handled in preConfirm -->
                        <option value="peak" ${slotData.type === 'peak' ? 'selected' : ''}>尖峰</option>
                        <option value="normal" ${slotData.type === 'normal' ? 'selected' : ''}>平段</option>
                        <option value="valley" ${slotData.type === 'valley' ? 'selected' : ''}>谷段</option>
                    </select>
                    <input type="text" class="form-control form-control-sm w-25"
                           placeholder="起始时间" pattern="([01]?[0-9]|2[0-3]):[0-5][0-9]" required value="${slotData.startTime || ''}">
                    <span>-</span>
                    <input type="text" class="form-control form-control-sm w-25"
                           placeholder="结束时间" pattern="([01]?[0-9]|2[0-3]):[0-5][0-9]" required value="${slotData.endTime || ''}">
                    <input type="number" class="form-control form-control-sm w-25"
                           placeholder="电价" step="0.01" min="0" required value="${slotData.price || ''}">
                    <span class="ms-1">元</span>
                    <button type="button" class="btn btn-sm btn-outline-danger p-1" onclick="strategyManager._removeTimeSlot(this)">
                        <i class="fas fa-minus"></i>
                    </button>
                </div>
                <div class="invalid-feedback d-none mt-1">时间格式错误 (HH:mm) 或结束时间必须大于开始时间</div>
                 <div class="invalid-feedback d-none mt-1">电价必须为非负数</div>
            </div>`;

        // Generate initial time slots HTML
        let timeSlotsHtml = '';
        if (strategy?.type === 'price' && strategy.parameters?.timeSlots?.length > 0) {
            timeSlotsHtml = strategy.parameters.timeSlots.map(generateTimeSlotHtml).join('');
        } else if (!isEditing) {
            timeSlotsHtml = generateTimeSlotHtml(); // Default empty slot for creation
        }

            Swal.fire({
            title: title,
                html: `
                <form id="tradingStrategyForm" class="text-start needs-validation" novalidate>
                        <div class="mb-3">
                            <label class="form-label">策略名称</label>
                        <input type="text" class="form-control" id="strategyName" value="${strategy?.name || ''}" required>
                         <div class="invalid-feedback">请输入策略名称</div>
                        </div>
                        <div class="mb-3">
                        <label class="form-label">交易类型</label>
                        <select class="form-select" id="tradingType" required ${isEditing ? 'disabled' : ''}> <!-- Disable type change on edit -->
                            <option value="price" ${strategy?.type === 'price' ? 'selected' : ''}>分时电价</option>
                            <option value="spot" ${strategy?.type === 'spot' ? 'selected' : ''}>现货交易</option>
                            <option value="auxiliary" ${strategy?.type === 'auxiliary' ? 'selected' : ''}>辅助服务</option>
                        </select>
                        </div>
                        <div class="mb-3">
                        <label class="form-label">策略描述</label>
                        <textarea class="form-control" id="strategyDescription" rows="2" required>${strategy?.description || ''}</textarea>
                         <div class="invalid-feedback">请输入策略描述</div>
                                </div>

                    <!-- Settings Containers -->
                    <div id="priceSettingsContainer" class="mb-3 ${strategy?.type !== 'price' && 'd-none'}">
                        <label class="form-label">分时电价设置</label>
                        <div id="priceTimeSlots">${timeSlotsHtml}</div>
                        <button type="button" class="btn btn-sm btn-outline-primary mt-2" onclick="strategyManager._addTimeSlot()">
                            <i class="fas fa-plus me-1"></i>添加时段
                        </button>
                        <div id="timeSlotsError" class="invalid-feedback d-none mt-1">时间段必须覆盖全天24小时且无重叠</div>
                                </div>
                    <div id="spotSettingsContainer" class="mb-3 ${strategy?.type !== 'spot' && 'd-none'}">
                        <label class="form-label">现货交易参数</label>
                         <input type="number" step="0.01" min="0" max="1" class="form-control" id="spotVolatilityThreshold"
                               placeholder="价格波动阈值 (如 0.1)" value="${strategy?.parameters?.priceVolatilityThreshold || ''}" required>
                        <div class="invalid-feedback">请输入有效的价格波动阈值 (0-1)</div>
                                </div>
                     <div id="auxiliarySettingsContainer" class="mb-3 ${strategy?.type !== 'auxiliary' && 'd-none'}">
                        <label class="form-label">辅助服务参数</label>
                        <input type="number" step="1" min="1" class="form-control" id="auxResponseTime"
                               placeholder="响应时间 (秒)" value="${strategy?.parameters?.responseTime || ''}" required>
                        <div class="invalid-feedback">请输入有效的响应时间 (秒)</div>
                            </div>
                    <!-- End Settings Containers -->

                    <div class="mb-3">
                        <label class="form-label">风险等级</label>
                        <select class="form-select" id="riskLevel" required>
                            <option value="low" ${strategy?.riskLevel === 'low' ? 'selected' : ''}>低风险</option>
                            <option value="medium" ${strategy?.riskLevel === 'medium' ? 'selected' : ''}>中等风险</option>
                            <option value="high" ${strategy?.riskLevel === 'high' ? 'selected' : ''}>高风险</option>
                        </select>
                        </div>
                        <div class="mb-3">
                        <label class="form-label">通用交易参数</label>
                        <div class="row g-2">
                            <div class="col-md-6">
                                <input type="number" step="0.1" min="0" class="form-control" id="minAmount"
                                       placeholder="最小交易量(MWh)" value="${strategy?.parameters?.minAmount || ''}" required>
                                 <div class="invalid-feedback">请输入有效的最小交易量</div>
                            </div>
                            <div class="col-md-6">
                                <input type="number" step="0.1" min="0" class="form-control" id="maxAmount"
                                       placeholder="最大交易量(MWh)" value="${strategy?.parameters?.maxAmount || ''}" required>
                                <div class="invalid-feedback">请输入有效的最大交易量</div>
                            </div>
                        </div>
                         <div id="amountError" class="invalid-feedback d-none mt-1">最大交易量必须大于或等于最小交易量</div>
                        </div>
                    </form>
                `,
                showCancelButton: true,
            confirmButtonText: isEditing ? '保存' : '创建',
                cancelButtonText: '取消',
            didOpen: () => {
                // Listener for trading type change (only relevant for creation)
                if (!isEditing) {
                     document.getElementById('tradingType').addEventListener('change', (e) => {
                        document.getElementById('priceSettingsContainer').classList.toggle('d-none', e.target.value !== 'price');
                        document.getElementById('spotSettingsContainer').classList.toggle('d-none', e.target.value !== 'spot');
                        document.getElementById('auxiliarySettingsContainer').classList.toggle('d-none', e.target.value !== 'auxiliary');
                        // Set required attributes based on visibility
                        this._updateRequiredAttributes('priceSettingsContainer', e.target.value === 'price');
                        this._updateRequiredAttributes('spotSettingsContainer', e.target.value === 'spot');
                        this._updateRequiredAttributes('auxiliarySettingsContainer', e.target.value === 'auxiliary');
                     });
                     // Initial required attribute setup for creation
                     const initialType = document.getElementById('tradingType').value;
                     this._updateRequiredAttributes('priceSettingsContainer', initialType === 'price');
                     this._updateRequiredAttributes('spotSettingsContainer', initialType === 'spot');
                     this._updateRequiredAttributes('auxiliarySettingsContainer', initialType === 'auxiliary');
                } else {
                    // Set required attributes for editing based on the existing strategy type
                     this._updateRequiredAttributes('priceSettingsContainer', strategy?.type === 'price');
                     this._updateRequiredAttributes('spotSettingsContainer', strategy?.type === 'spot');
                     this._updateRequiredAttributes('auxiliarySettingsContainer', strategy?.type === 'auxiliary');
                }

                 // Initialize validation states
                 const form = document.getElementById('tradingStrategyForm');
                 form.querySelectorAll('.is-invalid').forEach(el => el.classList.remove('is-invalid'));
                 form.querySelectorAll('.invalid-feedback.d-block').forEach(el => el.classList.replace('d-block', 'd-none'));

            },
                preConfirm: () => {
                const form = document.getElementById('tradingStrategyForm');
                let isValid = true;

                // Reset previous validation states
                form.querySelectorAll('.is-invalid').forEach(el => el.classList.remove('is-invalid'));
                form.querySelectorAll('.invalid-feedback.d-block').forEach(el => el.classList.replace('d-block', 'd-none'));

                // --- Basic Validation ---
                form.querySelectorAll('input[required]:not([disabled]), select[required]:not([disabled]), textarea[required]:not([disabled])').forEach(input => {
                    if (!input.value.trim()) {
                        input.classList.add('is-invalid');
                        const feedback = input.closest('div').querySelector('.invalid-feedback');
                        if(feedback) feedback.classList.replace('d-none', 'd-block');
                        isValid = false;
                    }
                });

                 // --- Amount Validation ---
                 const minAmountInput = document.getElementById('minAmount');
                 const maxAmountInput = document.getElementById('maxAmount');
                 const minAmount = parseFloat(minAmountInput.value);
                 const maxAmount = parseFloat(maxAmountInput.value);
                 const amountErrorDiv = document.getElementById('amountError');

                 if (isNaN(minAmount) || minAmount < 0) {
                     minAmountInput.classList.add('is-invalid');
                     minAmountInput.closest('div').querySelector('.invalid-feedback').classList.replace('d-none', 'd-block');
                     isValid = false;
                 }
                 if (isNaN(maxAmount) || maxAmount < 0) {
                     maxAmountInput.classList.add('is-invalid');
                     maxAmountInput.closest('div').querySelector('.invalid-feedback').classList.replace('d-none', 'd-block');
                     isValid = false;
                 }
                 if (!isNaN(minAmount) && !isNaN(maxAmount) && maxAmount < minAmount) {
                     maxAmountInput.classList.add('is-invalid');
                     amountErrorDiv.classList.replace('d-none', 'd-block');
                     isValid = false;
                 } else {
                    amountErrorDiv.classList.replace('d-block', 'd-none');
                 }

                const tradingType = document.getElementById('tradingType').value;
                let parameters = {
                     minAmount: minAmount,
                     maxAmount: maxAmount
                 };

                // --- Type-Specific Validation & Data Gathering ---
                if (tradingType === 'price') {
                    let parsedTimeSlots = [];
                    let totalMinutesCovered = 0;
                    const timeSlots = document.querySelectorAll('#priceTimeSlots .price-slot');
                    const timePattern = /^([01]?[0-9]|2[0-3]):[0-5][0-9]$/;
                    let timeSlotValid = true;

                    timeSlots.forEach(slot => {
                        const typeInput = slot.querySelector('select');
                        const startTimeInput = slot.querySelector('input[placeholder="起始时间"]');
                        const endTimeInput = slot.querySelector('input[placeholder="结束时间"]');
                        const priceInput = slot.querySelector('input[placeholder="电价"]');
                        const timeFeedback = slot.querySelector('.invalid-feedback:not([id])'); // Generic time feedback
                        const priceFeedback = slot.querySelector('.invalid-feedback:nth-of-type(2)'); // Price feedback

                         let currentSlotValid = true;

                        // Validate times
                        if (!timePattern.test(startTimeInput.value) || !timePattern.test(endTimeInput.value)) {
                            startTimeInput.classList.add('is-invalid');
                            endTimeInput.classList.add('is-invalid');
                             timeFeedback.textContent = '时间格式错误 (HH:mm)';
                            timeFeedback.classList.replace('d-none', 'd-block');
                            timeSlotValid = false;
                             currentSlotValid = false;
                        } else {
                             const startMins = this._timeToMinutes(startTimeInput.value);
                             const endMins = this._timeToMinutes(endTimeInput.value);
                             if (startMins >= endMins) {
                                 startTimeInput.classList.add('is-invalid');
                                 endTimeInput.classList.add('is-invalid');
                                 timeFeedback.textContent = '结束时间必须大于开始时间';
                                 timeFeedback.classList.replace('d-none', 'd-block');
                                 timeSlotValid = false;
                                 currentSlotValid = false;
                             }
                        }

                        // Validate price
                        const price = parseFloat(priceInput.value);
                        if (isNaN(price) || price < 0) {
                            priceInput.classList.add('is-invalid');
                            priceFeedback.classList.replace('d-none', 'd-block');
                            timeSlotValid = false;
                             currentSlotValid = false;
                        }

                        if (currentSlotValid) {
                            const startMins = this._timeToMinutes(startTimeInput.value);
                            const endMins = this._timeToMinutes(endTimeInput.value);
                            parsedTimeSlots.push({
                                type: typeInput.value,
                                startTime: startTimeInput.value,
                                endTime: endTimeInput.value,
                                price: price,
                                startMins: startMins, // For overlap check
                                endMins: endMins     // For overlap check
                            });
                            totalMinutesCovered += (endMins - startMins);
                        } else {
                            isValid = false; // Mark overall form as invalid if any slot fails
                        }
                    });

                    // Check for overlap and full day coverage if all individual slots are valid so far
                    const timeSlotsErrorDiv = document.getElementById('timeSlotsError');
                    if (timeSlotValid) {
                         // Sort by start time for easier overlap check
                        parsedTimeSlots.sort((a, b) => a.startMins - b.startMins);
                        let overlap = false;
                        for (let i = 0; i < parsedTimeSlots.length - 1; i++) {
                            if (parsedTimeSlots[i].endMins > parsedTimeSlots[i + 1].startMins) {
                                overlap = true;
                                break;
                            }
                        }

                        if (overlap) {
                            timeSlotsErrorDiv.textContent = '时间段存在重叠';
                            timeSlotsErrorDiv.classList.replace('d-none', 'd-block');
                            isValid = false;
                        } else if (totalMinutesCovered !== 1440) { // 24 * 60
                            timeSlotsErrorDiv.textContent = '时间段必须覆盖全天24小时';
                            timeSlotsErrorDiv.classList.replace('d-none', 'd-block');
                            isValid = false;
                        } else {
                             timeSlotsErrorDiv.classList.replace('d-block', 'd-none');
                        }
                    } else {
                         timeSlotsErrorDiv.classList.replace('d-block', 'd-none'); // Hide general error if specific slot errors exist
                         isValid = false; // Ensure form is invalid if individual slots failed
                    }

                    if(isValid){
                        // Only add cleaned slots if valid
                        parameters.timeSlots = parsedTimeSlots.map(({ startMins, endMins, ...rest }) => rest); // Remove temporary Mins properties
                    }

                } else if (tradingType === 'spot') {
                     const thresholdInput = document.getElementById('spotVolatilityThreshold');
                     const threshold = parseFloat(thresholdInput.value);
                     if (isNaN(threshold) || threshold < 0 || threshold > 1) {
                         thresholdInput.classList.add('is-invalid');
                         thresholdInput.closest('div').querySelector('.invalid-feedback').classList.replace('d-none', 'd-block');
                         isValid = false;
                     } else {
                         parameters.priceVolatilityThreshold = threshold;
                     }
                } else if (tradingType === 'auxiliary') {
                     const responseInput = document.getElementById('auxResponseTime');
                     const responseTime = parseInt(responseInput.value, 10);
                     if (isNaN(responseTime) || responseTime < 1) {
                         responseInput.classList.add('is-invalid');
                         responseInput.closest('div').querySelector('.invalid-feedback').classList.replace('d-none', 'd-block');
                         isValid = false;
                     } else {
                         parameters.responseTime = responseTime;
                     }
                }

                if (!isValid) {
                    return false; // Prevent Swal closing if validation failed
                }

                // Return collected data
                    return {
                    name: document.getElementById('strategyName').value,
                    type: tradingType, // Keep original type for editing lookup
                    description: document.getElementById('strategyDescription').value,
                    riskLevel: document.getElementById('riskLevel').value,
                    parameters: parameters
                    };
                }
            }).then((result) => {
                if (result.isConfirmed) {
                const data = result.value;
                if (isEditing) {
                    const index = this.strategies.trading.findIndex(s => s.id === strategyId);
                    if (index !== -1) {
                        // Merge parameters carefully
                        const existingParams = this.strategies.trading[index].parameters || {};
                        this.strategies.trading[index] = {
                            ...this.strategies.trading[index],
                            name: data.name,
                            description: data.description,
                            riskLevel: data.riskLevel,
                            parameters: { // Merge, ensuring type-specific params are updated/added
                                ...existingParams,
                                ...data.parameters
                            },
                            updateTime: new Date().toLocaleString('sv-SE')
                        };
                         // Recalculate profitRate based on new params if needed (example)
                         if(data.type === 'price') {
                             this.strategies.trading[index].profitRate = this._calculatePriceProfit(data.parameters.timeSlots) + '%';
                         }
                        Swal.fire('保存成功', '交易策略已更新', 'success');
                    } else {
                         Swal.fire('错误', '未找到要编辑的策略', 'error');
                         return;
                    }
                } else {
                    // Create new trading strategy
                     const newId = this._getNextId('trading');
                     const newStrategy = {
                        id: newId,
                        name: data.name,
                        type: data.type,
                        displayType: this._getTradingTypeDisplayName(data.type),
                        description: data.description,
                        riskLevel: data.riskLevel,
                        parameters: data.parameters,
                        profitRate: 'N/A', // Calculate initial profit rate if possible
                        successRate: 'N/A',
                        status: 'enabled',
                        updateTime: new Date().toLocaleString('sv-SE'),
                        version: 'v1.0.0',
                        author: '当前用户',
                        runCount: 0,
                        lastRunTime: 'N/A'
                     };
                     // Calculate initial profit rate
                     if (newStrategy.type === 'price' && newStrategy.parameters.timeSlots) {
                         newStrategy.profitRate = this._calculatePriceProfit(newStrategy.parameters.timeSlots) + '%';
                     }
                     this.strategies.trading.push(newStrategy);
                     Swal.fire('创建成功', '能源交易策略已创建', 'success');
                }
                this.refreshTables();
            }
        });
    },

    // Helper to update required attributes for trading modal sections
    _updateRequiredAttributes(containerId, isRequired) {
        const container = document.getElementById(containerId);
        if (container) {
            container.querySelectorAll('input, select').forEach(el => {
                if (isRequired) {
                    el.setAttribute('required', '');
                } else {
                    el.removeAttribute('required');
                     // Also reset validation state when hiding
                     el.classList.remove('is-invalid');
                     const feedback = el.closest('div').querySelector('.invalid-feedback');
                     if(feedback) feedback.classList.add('d-none');
                     const timeSlotsError = document.getElementById('timeSlotsError'); // Specific for time slots
                     if(timeSlotsError) timeSlotsError.classList.add('d-none');
                }
            });
        }
    },


    // Helper for adding a time slot row in the trading modal
    _addTimeSlot() {
        const container = document.getElementById('priceTimeSlots');
        const newSlotHtml = strategyManager._generateTimeSlotHtml(); // Use manager's context
        container.insertAdjacentHTML('beforeend', newSlotHtml);
    },

    // Helper for removing a time slot row in the trading modal
    _removeTimeSlot(button) {
        const slotsContainer = document.getElementById('priceTimeSlots');
        if (slotsContainer.querySelectorAll('.price-slot').length > 1) {
            button.closest('.price-slot').remove();
        } else {
                        Swal.fire({
                 text: '至少需要一个时间段',
                 icon: 'warning',
                 timer: 1500,
                            showConfirmButton: false
                        });
                    }
    },

     // Helper to generate time slot HTML string (moved inside manager)
    _generateTimeSlotHtml(slotData = { type: 'normal', startTime: '', endTime: '', price: '' }) {
        return `
            <div class="price-slot mb-2">
                 <div class="d-flex gap-2 align-items-center">
                    <select class="form-select form-select-sm w-25" required>
                        <option value="peak" ${slotData.type === 'peak' ? 'selected' : ''}>尖峰</option>
                        <option value="normal" ${slotData.type === 'normal' ? 'selected' : ''}>平段</option>
                        <option value="valley" ${slotData.type === 'valley' ? 'selected' : ''}>谷段</option>
                    </select>
                    <input type="text" class="form-control form-control-sm w-25"
                           placeholder="起始时间" pattern="([01]?[0-9]|2[0-3]):[0-5][0-9]" required value="${slotData.startTime || ''}">
                    <span>-</span>
                    <input type="text" class="form-control form-control-sm w-25"
                           placeholder="结束时间" pattern="([01]?[0-9]|2[0-3]):[0-5][0-9]" required value="${slotData.endTime || ''}">
                    <input type="number" class="form-control form-control-sm w-25"
                           placeholder="电价" step="0.01" min="0" required value="${slotData.price || ''}">
                    <span class="ms-1">元</span>
                    <button type="button" class="btn btn-sm btn-outline-danger p-1" onclick="strategyManager._removeTimeSlot(this)">
                        <i class="fas fa-minus"></i>
                    </button>
                 </div>
                 <div class="invalid-feedback d-none mt-1">时间格式错误 (HH:mm) 或结束时间必须大于开始时间</div>
                 <div class="invalid-feedback d-none mt-1">电价必须为非负数</div>
            </div>`;
    },


    // Example profit calculation for price strategy
     _calculatePriceProfit(timeSlots) {
         if (!timeSlots || timeSlots.length === 0) return 0;
         const prices = timeSlots.map(slot => slot.price);
         const maxPrice = Math.max(...prices);
         const minPrice = Math.min(...prices);
         // Very simple placeholder calculation
         return Math.round(((maxPrice - minPrice) / maxPrice) * 30 + 5); // Arbitrary calculation
     },

    // Edit strategy - updated signature
    editStrategy(type, id) {
        console.log(`Editing strategy - Type: ${type}, ID: ${id}`);
        const strategy = this.strategies[type]?.find(s => s.id === id);
        if (!strategy) {
            Swal.fire('错误', '未找到要编辑的策略', 'error');
            return;
        }

        if (type === 'trading') {
            this._showCreateTradingModal(id);
        } else {
            this._showCreateEditModal(type, id);
        }
    },

    // Duplicate strategy - updated signature
    duplicateStrategy(type, id) {
        console.log(`Duplicating strategy - Type: ${type}, ID: ${id}`);
        const strategy = this.strategies[type]?.find(s => s.id === id);
        if (!strategy) {
            Swal.fire('错误', '未找到要复制的策略', 'error');
            return;
        }

        // Deep copy might be needed for parameters
        const newStrategy = JSON.parse(JSON.stringify(strategy));

        newStrategy.id = this._getNextId(type);
        newStrategy.name = `${strategy.name} (副本)`;
        newStrategy.type = 'custom'; // Duplicated strategies are custom
        newStrategy.updateTime = new Date().toLocaleString('sv-SE');
        newStrategy.version = 'v1.0.0';
        newStrategy.runCount = 0;
        newStrategy.lastRunTime = 'N/A';
        newStrategy.author = '当前用户'; // Or actual user
        // Reset performance metrics
        newStrategy.accuracy = 'N/A';
        newStrategy.profitRate = 'N/A';
        newStrategy.successRate = 'N/A';

        this.strategies[type].push(newStrategy);
        this.refreshTables();

            Swal.fire({
                title: '复制成功',
            text: '策略已复制，您可以进行编辑',
                icon: 'success',
                timer: 2000,
                showConfirmButton: false
            });
        },

    // Delete strategy - updated signature
    deleteStrategy(type, id) {
        console.log(`Deleting strategy - Type: ${type}, ID: ${id}`);
        const strategyIndex = this.strategies[type]?.findIndex(s => s.id === id);
        if (strategyIndex === -1 || !this.strategies[type]) {
            Swal.fire('错误', '未找到要删除的策略', 'error');
            return;
        }
        const strategy = this.strategies[type][strategyIndex];

            Swal.fire({
                title: '确认删除',
                html: `
                    <div class="text-start">
                        <p class="mb-2">是否删除以下策略：</p>
                        <div class="alert alert-warning">
                            <h6 class="alert-heading">${strategy.name}</h6>
                            <p class="mb-0 small">${strategy.description}</p>
                        </div>
                        <p class="mb-0 text-danger">
                            <i class="fas fa-exclamation-triangle me-1"></i>
                            此操作不可恢复，请谨慎操作
                        </p>
                    </div>
                `,
                icon: 'warning',
                showCancelButton: true,
                confirmButtonText: '删除',
                cancelButtonText: '取消',
                confirmButtonColor: '#dc3545'
            }).then((result) => {
                if (result.isConfirmed) {
                this.strategies[type].splice(strategyIndex, 1);
                this.refreshTables();
                        
                        Swal.fire({
                            title: '删除成功',
                            text: '策略已被删除',
                            icon: 'success',
                            timer: 2000,
                            showConfirmButton: false
                        });
                }
            });
        },

    // Toggle strategy status - updated signature
    toggleStrategy(type, id, checkbox) {
        console.log(`Toggling strategy status - Type: ${type}, ID: ${id}`);
        const strategy = this.strategies[type]?.find(s => s.id === id);
        if (!strategy) {
             checkbox.checked = !checkbox.checked; // Revert UI
             Swal.fire('错误', '未找到要操作的策略', 'error');
             return;
        }

        const newStatus = checkbox.checked ? 'enabled' : 'disabled';
        const actionText = newStatus === 'enabled' ? '启用' : '禁用';

        // Update status immediately for responsiveness? Or wait for confirmation?
        // Let's wait for confirmation to be safer. Revert UI if cancelled.
        checkbox.checked = !checkbox.checked; // Temporarily revert

            Swal.fire({
            title: `确认${actionText}`,
            text: `是否${actionText}策略 "${strategy.name}"?`,
            icon: 'question',
            showCancelButton: true,
            confirmButtonText: actionText,
            cancelButtonText: '取消'
        }).then((result) => {
            if (result.isConfirmed) {
                strategy.status = newStatus;
                strategy.updateTime = new Date().toLocaleString('sv-SE');
                checkbox.checked = (newStatus === 'enabled'); // Set final state
                this.refreshTables(); // Refresh to update badge if necessary
                 Swal.fire({
                     title: `${actionText}成功`,
                     icon: 'success',
                     timer: 1500,
                     showConfirmButton: false
                 });
            }
             // If cancelled, the checkbox remains reverted.
        });
    },


    // View strategy details (using Swal for now) - updated signature
    viewStrategyDetail(type, id) {
        console.log(`Viewing strategy details - Type: ${type}, ID: ${id}`);
        const strategy = this.strategies[type]?.find(s => s.id === id);
        if (!strategy) {
            Swal.fire('错误', '未找到策略详情', 'error');
            return;
        }

        // Helper to format parameter/constraint/objective values
        const formatValue = (value) => {
            if (Array.isArray(value)) {
                if (value.length === 0) return '<span class="text-muted">无</span>';
                return `<ul class="list-unstyled ms-3 mb-0 small">${value.map(item => `<li>${item}</li>`).join('')}</ul>`;
            } else if (typeof value === 'object' && value !== null) {
                // Special handling for timeSlots for better display
                if (Array.isArray(value.timeSlots)) {
                    let tsHtml = '<ul class="list-unstyled ms-3 mb-0 small">';
                    value.timeSlots.forEach(slot => {
                        tsHtml += `<li>${slot.startTime}-${slot.endTime} (${slot.type}): ${slot.price}元</li>`;
                    });
                    tsHtml += '</ul>';
                    // Display other parameters besides timeSlots
                    const otherParams = { ...value };
                    delete otherParams.timeSlots;
                    let otherParamsHtml = '';
                    if(Object.keys(otherParams).length > 0) {
                        otherParamsHtml = `<pre class="bg-light p-2 rounded small mb-0 mt-1"><code>${JSON.stringify(otherParams, null, 2)}</code></pre>`;
                    }
                    return tsHtml + otherParamsHtml;
                } else {
                    return `<pre class="bg-light p-2 rounded small mb-0"><code>${JSON.stringify(value, null, 2)}</code></pre>`;
                }
            } else if (typeof value === 'boolean') {
                return value ? '<span class="text-success">是</span>' : '<span class="text-danger">否</span>';
            } else if (value === null || value === undefined || value === '') {
                return '<span class="text-muted">N/A</span>';
            }
            return String(value); // Ensure it's a string
        };

        // Helper to build table rows for objects
        const buildTableRows = (obj, titleMap = {}) => {
            if (!obj || typeof obj !== 'object' || Object.keys(obj).length === 0) {
                return '<tr><td colspan="2" class="text-muted text-center small">无</td></tr>';
            }
            return Object.entries(obj)
                .map(([key, value]) => {
                     // Skip complex objects if they were handled by formatValue (like parameters.timeSlots)
                     if (key === 'timeSlots' && typeof value === 'object') return '';
                     return `<tr><th width="40%">${titleMap[key] || key}</th><td>${formatValue(value)}</td></tr>`;
                })
                .join('');
        };

        // --- Generate Details HTML dynamically --- //
        let detailsHtml = `<div class="accordion" id="strategyDetailAccordion">`;

        // --- 1. Basic Info --- //
        detailsHtml += `
                        <div class="accordion-item">
                            <h2 class="accordion-header">
                    <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#basicInfo">基本信息</button>
                            </h2>
                            <div id="basicInfo" class="accordion-collapse collapse show" data-bs-parent="#strategyDetailAccordion">
                                <div class="accordion-body">
                        <table class="table table-sm table-borderless">
                            <tr><th width="30%">策略名称</th><td>${strategy.name}</td></tr>
                            <tr><th>策略类型</th><td>${this._getTypeDisplayName(type)} / ${strategy.displayType || strategy.type}</td></tr>
                            <tr><th>策略描述</th><td><small>${strategy.description || 'N/A'}</small></td></tr>
                            <tr><th>当前版本</th><td>${strategy.version || 'N/A'}</td></tr>
                            <tr><th>创建/修改人</th><td>${strategy.author || 'N/A'}</td></tr>
                            <tr><th>最后更新</th><td>${strategy.updateTime || 'N/A'}</td></tr>
                            <tr><th>运行状态</th><td>
                                                <span class="badge bg-${strategy.status === 'enabled' ? 'success' : 'danger'}">
                                                    ${strategy.status === 'enabled' ? '已启用' : '已禁用'}
                                                </span>
                            </td></tr>
                                    </table>
                                </div>
                            </div>
            </div>`;

        // --- 2. Configuration --- //
        detailsHtml += `
                        <div class="accordion-item">
                            <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#configDetails">配置详情</button>
                            </h2>
                <div id="configDetails" class="accordion-collapse collapse" data-bs-parent="#strategyDetailAccordion">
                                <div class="accordion-body">
                        <div class="mb-2"><strong>算法:</strong>
                            ${(strategy.algorithms || []).map(a => `<span class="badge bg-primary ms-1">${a}</span>`).join('') || '<span class="text-muted small">N/A</span>'}
                                        </div>
                        <div class="mb-2"><strong>数据源/因素:</strong>
                            ${(strategy.dataSource || strategy.factors || []).map(d => `<span class="badge bg-info ms-1">${d}</span>`).join('') || '<span class="text-muted small">N/A</span>'}
                                    </div>
                         <div class="mb-2"><strong>特征工程:</strong>
                            ${(strategy.features || []).map(f => `<span class="badge bg-secondary ms-1">${f}</span>`).join('') || '<span class="text-muted small">N/A</span>'}
                        </div>`;

        // Type-specific configuration display
        if (type === 'renewable') {
            detailsHtml += `<div class="mb-2"><strong>时间尺度:</strong> ${strategy.timeScale || 'N/A'}</div>`;
        } else if (type === 'load') {
            detailsHtml += `<div class="mb-2"><strong>预测周期:</strong> ${strategy.predictionCycle || 'N/A'}</div>`;
        } else if (type === 'charging') {
            detailsHtml += `<div class="mb-2"><strong>预测范围:</strong> ${strategy.predictionRange || 'N/A'}</div>`;
        } else if (type === 'trading') {
             detailsHtml += `<div class="mb-2"><strong>交易类型:</strong> ${strategy.displayType || 'N/A'}</div>`;
             detailsHtml += `<div class="mb-2"><strong>风险等级:</strong> <span class="badge bg-${this.getRiskColor(strategy.riskLevel)}">${this.getRiskName(strategy.riskLevel)}</span></div>`;
        }
        detailsHtml += `</div></div></div>`; // Close config accordion item

        // --- 3. Parameters --- //
        if (strategy.parameters && Object.keys(strategy.parameters).length > 0) {
             detailsHtml += `
                        <div class="accordion-item">
                            <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#paramDetails">运行参数</button>
                            </h2>
                    <div id="paramDetails" class="accordion-collapse collapse" data-bs-parent="#strategyDetailAccordion">
                                <div class="accordion-body">
                            <table class="table table-sm table-bordered table-hover small">
                                <tbody>
                                    ${buildTableRows(strategy.parameters)}
                                </tbody>
                             </table>
                                                        </div>
                    </div>
                </div>`;
        }

        // --- 4. Constraints --- //
        if (strategy.constraints && Object.keys(strategy.constraints).length > 0) {
             detailsHtml += `
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#constraintDetails">约束条件</button>
                    </h2>
                    <div id="constraintDetails" class="accordion-collapse collapse" data-bs-parent="#strategyDetailAccordion">
                        <div class="accordion-body">
                             <table class="table table-sm table-bordered table-hover small">
                                 <tbody>
                                     ${buildTableRows(strategy.constraints)}
                                 </tbody>
                                    </table>
                                </div>
                            </div>
                </div>`;
        }

        // --- 5. Objectives --- //
        if (strategy.objectives && strategy.objectives.length > 0) {
            detailsHtml += `
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#objectiveDetails">策略目标</button>
                    </h2>
                    <div id="objectiveDetails" class="accordion-collapse collapse" data-bs-parent="#strategyDetailAccordion">
                        <div class="accordion-body">
                             ${formatValue(strategy.objectives)}
                        </div>
                        </div>
                </div>`;
        }

        // --- 6. Performance & Stats --- //
        detailsHtml += `
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#perfStats">性能统计</button>
                </h2>
                <div id="perfStats" class="accordion-collapse collapse" data-bs-parent="#strategyDetailAccordion">
                    <div class="accordion-body">
                         <table class="table table-sm table-borderless">
                             <tr><th width="30%">累计运行次数</th><td>${strategy.runCount ?? 'N/A'} 次</td></tr>
                             <tr><th>最近运行时间</th><td>${strategy.lastRunTime || 'N/A'}</td></tr>`;

        // Add primary performance metric display first
        if (type === 'renewable' || type === 'load' || type === 'charging') {
             detailsHtml += `<tr><th>准确率(主要)</th><td><strong class="text-${this.getAccuracyColor(strategy.accuracy)}">${strategy.accuracy || 'N/A'}</strong></td></tr>`;
        } else if (type === 'trading') {
             detailsHtml += `<tr><th>预期收益率</th><td><strong class="text-${this.getProfitColor(strategy.profitRate)}">${strategy.profitRate || 'N/A'}</strong></td></tr>`;
             detailsHtml += `<tr><th>交易成功率</th><td><strong>${strategy.successRate || 'N/A'}</strong></td></tr>`;
        }

        // Add detailed performance metrics table if available
        if (strategy.performanceMetrics && Object.keys(strategy.performanceMetrics).length > 0) {
            detailsHtml += `
                <tr><td colspan="2">
                    <strong class="d-block mb-2">详细指标:</strong>
                    <table class="table table-sm table-bordered table-hover small">
                        <tbody>
                           ${buildTableRows(strategy.performanceMetrics)}
                        </tbody>
                    </table>
                </td></tr>`;
        } else {
            detailsHtml += `<tr><th>详细指标</th><td><span class="text-muted">N/A</span></td></tr>`;
        }

        detailsHtml += `
                        </table>
                    </div>
                </div>
            </div>`; // Close Performance accordion item

         detailsHtml += `</div>`; // Close the main accordion div

                    Swal.fire({
            title: '策略详情: ' + strategy.name,
            html: detailsHtml,
            width: '800px',
            showCloseButton: true,
            showConfirmButton: false,
            customClass: {
                // Ensure modal content is scrollable if it gets too long
                 popup: 'swal2-popup-scrollable',
                 htmlContainer: 'swal2-html-container-scrollable'
                }
            });
        },

    // Refresh all strategy tables
    refreshTables() {
        this._refreshTable('renewable', '#renewableStrategyTable', this._renderRenewableRow);
        this._refreshTable('load', '#loadStrategyTable', this._renderLoadRow);
        this._refreshTable('charging', '#chargingStrategyTable', this._renderChargingRow);
        this._refreshTable('trading', '#tradingStrategyTable', this._renderTradingRow);
    },

    // Generic table refresh helper
    _refreshTable(type, tableSelector, rowRenderer) {
        const tbody = document.querySelector(tableSelector + ' tbody');
        if (!tbody) {
            console.warn(`Table body not found for selector: ${tableSelector}`);
            return;
        }
        // Ensure rowRenderer is bound to the strategyManager context
        tbody.innerHTML = (this.strategies[type] || []).map(strategy => rowRenderer.call(this, strategy)).join('');
    },

    // --- Row Rendering Functions ---
    _renderRenewableRow(strategy) {
        return `
                    <tr>
                        <td>
                    <a href="javascript:void(0)" onclick="strategyManager.viewStrategyDetail('renewable', ${strategy.id})" class="fw-bold">${strategy.name}</a>
                    <div class="small text-muted">${strategy.description}</div>
                        </td>
                        <td>
                            <span class="badge bg-${strategy.type === 'wind' ? 'info' : strategy.type === 'solar' ? 'warning' : 'primary'}">
                        ${strategy.displayType || strategy.type}
                            </span>
                        </td>
                        <td>${strategy.timeScale}</td>
                        <td>${strategy.updateTime}</td>
                        <td>
                    <span class="badge bg-${strategy.status === 'enabled' ? 'success' : 'danger'} me-1">
                                    ${strategy.status === 'enabled' ? '已启用' : '已禁用'}
                                </span>
                    <span class="text-${this.getAccuracyColor(strategy.accuracy)}">${strategy.accuracy}</span>
                        </td>
                        <td>
                            <div class="btn-group">
                                <button class="btn btn-sm btn-outline-primary" onclick="strategyManager.editStrategy('renewable', ${strategy.id})">
                            <i class="fas fa-edit"></i> <span class="d-none d-md-inline">编辑</span>
                                </button>
                                <button class="btn btn-sm btn-outline-info" onclick="strategyManager.duplicateStrategy('renewable', ${strategy.id})">
                            <i class="fas fa-copy"></i> <span class="d-none d-md-inline">复制</span>
                                </button>
                                <button class="btn btn-sm btn-outline-danger" onclick="strategyManager.deleteStrategy('renewable', ${strategy.id})">
                            <i class="fas fa-trash"></i> <span class="d-none d-md-inline">删除</span>
                                </button>
                            </div>
                        </td>
                    </tr>
        `;
    },

    _renderLoadRow(strategy) {
         return `
                    <tr>
                        <td>
                     <a href="javascript:void(0)" onclick="strategyManager.viewStrategyDetail('load', ${strategy.id})" class="fw-bold">${strategy.name}</a>
                     <div class="small text-muted">${strategy.description}</div>
                        </td>
                        <td>${strategy.predictionCycle}</td>
                        <td>
                            <div class="d-flex flex-wrap gap-1">
                         ${(strategy.factors || []).map(factor => `<span class="badge bg-info">${factor}</span>`).join('')}
                            </div>
                        </td>
                        <td>${strategy.updateTime}</td>
                        <td>
                     <span class="badge bg-${strategy.status === 'enabled' ? 'success' : 'danger'} me-1">
                                    ${strategy.status === 'enabled' ? '已启用' : '已禁用'}
                                </span>
                     <span class="text-${this.getAccuracyColor(strategy.accuracy)}">${strategy.accuracy}</span>
                        </td>
                        <td>
                            <div class="btn-group">
                                <button class="btn btn-sm btn-outline-primary" onclick="strategyManager.editStrategy('load', ${strategy.id})">
                            <i class="fas fa-edit"></i> <span class="d-none d-md-inline">编辑</span>
                                </button>
                                <button class="btn btn-sm btn-outline-info" onclick="strategyManager.duplicateStrategy('load', ${strategy.id})">
                            <i class="fas fa-copy"></i> <span class="d-none d-md-inline">复制</span>
                                </button>
                                <button class="btn btn-sm btn-outline-danger" onclick="strategyManager.deleteStrategy('load', ${strategy.id})">
                            <i class="fas fa-trash"></i> <span class="d-none d-md-inline">删除</span>
                                </button>
                            </div>
                        </td>
                    </tr>
         `;
    },

    _renderChargingRow(strategy) {
         return `
                    <tr>
                        <td>
                     <a href="javascript:void(0)" onclick="strategyManager.viewStrategyDetail('charging', ${strategy.id})" class="fw-bold">${strategy.name}</a>
                     <div class="small text-muted">${strategy.description}</div>
                        </td>
                        <td>${strategy.predictionRange}</td>
                        <td>
                            <div class="d-flex flex-wrap gap-1">
                         ${(strategy.factors || []).map(factor => `<span class="badge bg-info">${factor}</span>`).join('')}
                            </div>
                        </td>
                        <td>${strategy.updateTime}</td>
                        <td>
                     <span class="badge bg-${strategy.status === 'enabled' ? 'success' : 'danger'} me-1">
                                    ${strategy.status === 'enabled' ? '已启用' : '已禁用'}
                                </span>
                     <span class="text-${this.getAccuracyColor(strategy.accuracy)}">${strategy.accuracy}</span>
                        </td>
                        <td>
                            <div class="btn-group">
                                <button class="btn btn-sm btn-outline-primary" onclick="strategyManager.editStrategy('charging', ${strategy.id})">
                            <i class="fas fa-edit"></i> <span class="d-none d-md-inline">编辑</span>
                                </button>
                                <button class="btn btn-sm btn-outline-info" onclick="strategyManager.duplicateStrategy('charging', ${strategy.id})">
                            <i class="fas fa-copy"></i> <span class="d-none d-md-inline">复制</span>
                                </button>
                                <button class="btn btn-sm btn-outline-danger" onclick="strategyManager.deleteStrategy('charging', ${strategy.id})">
                            <i class="fas fa-trash"></i> <span class="d-none d-md-inline">删除</span>
                                </button>
                            </div>
                        </td>
                    </tr>
         `;
    },

    _renderTradingRow(strategy) {
        return `
                    <tr>
                        <td>
                    <a href="javascript:void(0)" onclick="strategyManager.viewStrategyDetail('trading', ${strategy.id})" class="fw-bold">${strategy.name}</a>
                     <div class="small text-muted">${strategy.description}</div>
                        </td>
                        <td><span class="badge bg-${this.getTypeColor(strategy.type)}">${this.getTypeName(strategy.type)}</span></td>
                        <td><span class="text-${this.getProfitColor(strategy.profitRate)}">${strategy.profitRate}</span></td>
                        <td><span class="badge bg-${this.getRiskColor(strategy.riskLevel)}">${this.getRiskName(strategy.riskLevel)}</span></td>
                        <td>${strategy.updateTime}</td>
                        <td>
                    <div class="form-check form-switch d-inline-block align-middle">
                        <input class="form-check-input" type="checkbox" role="switch"
                               id="toggleTrading${strategy.id}"
                                       ${strategy.status === 'enabled' ? 'checked' : ''}
                                       onchange="strategyManager.toggleStrategy('trading', ${strategy.id}, this)">
                        <label class="form-check-label" for="toggleTrading${strategy.id}"></label>
                            </div>
                        </td>
                        <td>
                            <div class="btn-group">
                                <button class="btn btn-sm btn-outline-primary" onclick="strategyManager.editStrategy('trading', ${strategy.id})">
                            <i class="fas fa-edit"></i> <span class="d-none d-md-inline">编辑</span>
                                </button>
                                <button class="btn btn-sm btn-outline-info" onclick="strategyManager.duplicateStrategy('trading', ${strategy.id})">
                            <i class="fas fa-copy"></i> <span class="d-none d-md-inline">复制</span>
                                </button>
                                <button class="btn btn-sm btn-outline-danger" onclick="strategyManager.deleteStrategy('trading', ${strategy.id})">
                            <i class="fas fa-trash"></i> <span class="d-none d-md-inline">删除</span>
                                </button>
                            </div>
                        </td>
                    </tr>
        `;
        },


    // --- Helper Functions (moved inside) ---
        getAccuracyColor(accuracy) {
        if (!accuracy || accuracy === 'N/A') return 'muted';
        const value = parseFloat(accuracy.replace('%',''));
        if (isNaN(value)) return 'muted';
            if (value >= 90) return 'success';
            if (value >= 80) return 'warning';
            return 'danger';
        },

        getTypeColor(type) {
        const colors = { price: 'info', spot: 'primary', bilateral: 'success', auxiliary: 'warning' };
            return colors[type] || 'secondary';
        },

    getTypeName(type) { // Keep for potential fallback display
        const names = { price: '分时电价', spot: '现货交易', bilateral: '双边交易', auxiliary: '辅助服务' };
            return names[type] || '未知类型';
    },

     _getTypeDisplayName(type) { // For modal titles etc.
         const names = { renewable: '风光发电', load: '负荷预测', charging: '充电预测', trading: '能源交易' };
         return names[type] || '未知类型';
     },
     _getTradingTypeDisplayName(type) {
         const names = { price: '分时电价', spot: '现货市场', auxiliary: '辅助服务' };
         return names[type] || '未知';
        },

        getProfitColor(profit) {
            const value = parseFloat(profit);
            if (value >= 20) return 'success';
            if (value >= 10) return 'primary';
            return 'danger';
        },

        getRiskColor(risk) {
        const colors = { low: 'success', medium: 'warning', high: 'danger' };
            return colors[risk] || 'secondary';
        },

        getRiskName(risk) {
        const names = { low: '低风险', medium: '中等风险', high: '高风险' };
            return names[risk] || '未知风险';
        },

     _timeToMinutes(time) {
         if(!time) return 0;
         const parts = time.split(':');
         if(parts.length !== 2) return 0;
         const hours = parseInt(parts[0], 10);
         const minutes = parseInt(parts[1], 10);
         if(isNaN(hours) || isNaN(minutes)) return 0;
        return hours * 60 + minutes;
     },

     // Initialization function for the manager
     init() {
         console.log("Initializing Strategy Manager...");
         // Ensure DOM is ready before refreshing tables
         if (document.readyState === 'loading') {
             document.addEventListener('DOMContentLoaded', () => {
                 this.refreshTables();
                 this._attachEventListeners();
             });
         } else {
             this.refreshTables();
             this._attachEventListeners();
         }
     },

     _attachEventListeners() {
         // Attach listeners for the 'New Strategy' dropdown items
         const newStrategyDropdown = document.getElementById('newStrategyDropdown');
         if (newStrategyDropdown) {
             const dropdownItems = newStrategyDropdown.nextElementSibling.querySelectorAll('.dropdown-item[data-type]');
        dropdownItems.forEach(item => {
                 // Remove previous listeners if any (important for potential re-init)
                 item.onclick = null; // Clear potential old onclick
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const type = e.target.getAttribute('data-type');
                if (type) {
                         this.createStrategy(type);
                }
            });
        });
              console.log("New Strategy dropdown listeners attached.");
            } else {
             console.warn("New Strategy dropdown not found.");
         }

         // Add listeners for tab changes if needed (e.g., to resize charts)
         const tabs = document.querySelectorAll('#strategyTabs .nav-link');
        tabs.forEach(tab => {
             tab.addEventListener('shown.bs.tab', event => {
                 console.log(`Tab shown: ${event.target.id}`);
                 // Example: resize charts if they exist in the shown tab
                 // window.dispatchEvent(new Event('resize'));
            });
        });
     }

};

// Initialize the strategy manager
strategyManager.init();