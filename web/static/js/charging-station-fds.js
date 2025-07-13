/**
 * 充电桩故障诊断系统
 */
class FaultDiagnosisSystem {
    constructor() {
        // 故障规则库
        this.faultRules = new Map();
        // 故障历史记录
        this.faultHistory = new Map();
        // 诊断模型
        this.diagnosisModels = new Map();
        // 初始化
        this.initialize();
    }

    initialize() {
        this.initializeFaultRules();
        this.initializeDiagnosisModels();
    }

    /**
     * 初始化故障规则库
     */
    initializeFaultRules() {
        // 电力系统故障规则
        this.faultRules.set('power', [
            {
                id: 'P001',
                name: '输出过压',
                conditions: {
                    outputVoltage: (value) => value > 1.1 * this.RATED_VOLTAGE,
                    duration: (value) => value > 5 // 持续5秒以上
                },
                severity: 'critical',
                category: 'power',
                suggestedActions: [
                    '立即断开输出',
                    '检查电压调节器',
                    '检查输出滤波电路'
                ]
            },
            {
                id: 'P002',
                name: '输出欠压',
                conditions: {
                    outputVoltage: (value) => value < 0.9 * this.RATED_VOLTAGE,
                    duration: (value) => value > 5
                },
                severity: 'warning',
                category: 'power',
                suggestedActions: [
                    '检查输入电源',
                    '检查电压调节回路',
                    '检查负载状态'
                ]
            },
            {
                id: 'P003',
                name: '过流保护',
                conditions: {
                    outputCurrent: (value) => value > 1.2 * this.RATED_CURRENT,
                    duration: (value) => value > 3
                },
                severity: 'critical',
                category: 'power',
                suggestedActions: [
                    '立即断开输出',
                    '检查负载是否短路',
                    '检查电流采样电路'
                ]
            }
        ]);

        // 通信系统故障规则
        this.faultRules.set('communication', [
            {
                id: 'C001',
                name: '通信中断',
                conditions: {
                    connectionStatus: (value) => value === 'disconnected',
                    duration: (value) => value > 30
                },
                severity: 'warning',
                category: 'communication',
                suggestedActions: [
                    '检查网络连接',
                    '重置通信模块',
                    '检查通信协议配置'
                ]
            },
            {
                id: 'C002',
                name: '数据异常',
                conditions: {
                    packetLoss: (value) => value > 0.1,
                    errorRate: (value) => value > 0.05
                },
                severity: 'notice',
                category: 'communication',
                suggestedActions: [
                    '检查数据校验',
                    '分析通信日志',
                    '更新通信参数'
                ]
            }
        ]);

        // 温度控制故障规则
        this.faultRules.set('temperature', [
            {
                id: 'T001',
                name: '充电接口过温',
                conditions: {
                    interfaceTemp: (value) => value > 85,
                    duration: (value) => value > 10
                },
                severity: 'critical',
                category: 'temperature',
                suggestedActions: [
                    '立即停止充电',
                    '启动强制散热',
                    '检查散热系统'
                ]
            },
            {
                id: 'T002',
                name: '散热器过温',
                conditions: {
                    radiatorTemp: (value) => value > 70,
                    fanStatus: (value) => value !== 'normal'
                },
                severity: 'warning',
                category: 'temperature',
                suggestedActions: [
                    '降低充电功率',
                    '检查风扇运行',
                    '清理散热器'
                ]
            }
        ]);

        // 充电控制故障规则
        this.faultRules.set('charging', [
            {
                id: 'CH001',
                name: '充电握手失败',
                conditions: {
                    handshakeStatus: (value) => value === 'failed',
                    retryCount: (value) => value >= 3
                },
                severity: 'warning',
                category: 'charging',
                suggestedActions: [
                    '检查通信协议',
                    '检查车辆兼容性',
                    '重置充电控制器'
                ]
            },
            {
                id: 'CH002',
                name: '充电中断',
                conditions: {
                    chargingStatus: (value) => value === 'interrupted',
                    voltage: (value) => value < 0.5 * this.RATED_VOLTAGE
                },
                severity: 'warning',
                category: 'charging',
                suggestedActions: [
                    '检查车辆连接',
                    '分析中断原因',
                    '尝试重新启动充电'
                ]
            }
        ]);
    }

    /**
     * 初始化诊断模型
     */
    initializeDiagnosisModels() {
        // 基于规则的诊断模型
        this.diagnosisModels.set('rule-based', {
            name: '规则诊断',
            diagnose: (data) => this.ruleBasedDiagnosis(data)
        });

        // 基于统计的诊断模型
        this.diagnosisModels.set('statistical', {
            name: '统计诊断',
            diagnose: (data) => this.statisticalDiagnosis(data)
        });

        // 基于模式的诊断模型
        this.diagnosisModels.set('pattern', {
            name: '模式诊断',
            diagnose: (data) => this.patternDiagnosis(data)
        });
    }

    /**
     * 执行故障诊断
     * @param {string} stationId 充电站ID
     * @param {string} portId 充电端口ID
     * @param {Object} data 诊断数据
     */
    async diagnoseFault(stationId, portId, data) {
        try {
            // 1. 数据预处理
            const processedData = this.preprocessData(data);

            // 2. 多模型诊断
            const diagnoseResults = await Promise.all([
                this.diagnosisModels.get('rule-based').diagnose(processedData),
                this.diagnosisModels.get('statistical').diagnose(processedData),
                this.diagnosisModels.get('pattern').diagnose(processedData)
            ]);

            // 3. 结果融合
            const fusedResult = this.fuseResults(diagnoseResults);

            // 4. 可信度评估
            const reliability = this.assessReliability(fusedResult, processedData);

            // 5. 生成诊断报告
            const report = this.generateDiagnosisReport(stationId, portId, fusedResult, reliability);

            // 6. 更新故障历史
            this.updateFaultHistory(stationId, portId, report);

            return report;
        } catch (error) {
            console.error('故障诊断失败:', error);
            throw error;
        }
    }

    /**
     * 数据预处理
     * @private
     */
    preprocessData(data) {
        return {
            ...data,
            timestamp: new Date(),
            // 添加数据验证
            isValid: this.validateData(data),
            // 添加数据标准化
            normalized: this.normalizeData(data),
            // 添加特征提取
            features: this.extractFeatures(data)
        };
    }

    /**
     * 基于规则的诊断
     * @private
     */
    ruleBasedDiagnosis(data) {
        const results = [];

        // 遍历所有故障规则
        for (const [category, rules] of this.faultRules) {
            for (const rule of rules) {
                // 检查是否满足故障条件
                const conditionsMet = Object.entries(rule.conditions)
                    .every(([key, condition]) => condition(data[key]));

                if (conditionsMet) {
                    results.push({
                        ruleId: rule.id,
                        category: rule.category,
                        name: rule.name,
                        severity: rule.severity,
                        confidence: this.calculateRuleConfidence(rule, data),
                        suggestedActions: rule.suggestedActions
                    });
                }
            }
        }

        return results;
    }

    /**
     * 基于统计的诊断
     * @private
     */
    statisticalDiagnosis(data) {
        const results = [];

        // 1. 计算关键指标的统计特征
        const stats = {
            voltage: this.calculateStatistics(data.voltageHistory),
            current: this.calculateStatistics(data.currentHistory),
            temperature: this.calculateStatistics(data.temperatureHistory)
        };

        // 2. 检测异常值
        const anomalies = this.detectAnomalies(data, stats);

        // 3. 分析趋势
        const trends = this.analyzeTrends(data.history);

        // 4. 生成诊断结果
        if (anomalies.length > 0 || trends.some(t => t.type === 'negative')) {
            results.push({
                method: 'statistical',
                anomalies,
                trends,
                confidence: this.calculateStatisticalConfidence(anomalies, trends)
            });
        }

        return results;
    }

    /**
     * 基于模式的诊断
     * @private
     */
    patternDiagnosis(data) {
        const results = [];

        // 1. 提取特征模式
        const patterns = this.extractPatterns(data);

        // 2. 匹配已知故障模式
        const matches = this.matchKnownPatterns(patterns);

        // 3. 评估模式相似度
        const similarities = this.calculatePatternSimilarities(matches);

        // 4. 生成诊断结果
        if (similarities.length > 0) {
            results.push({
                method: 'pattern',
                matches: similarities.filter(s => s.similarity > 0.8),
                confidence: this.calculatePatternConfidence(similarities)
            });
        }

        return results;
    }

    /**
     * 结果融合
     * @private
     */
    fuseResults(results) {
        // 1. 权重分配
        const weights = {
            'rule-based': 0.4,
            'statistical': 0.3,
            'pattern': 0.3
        };

        // 2. 结果归一化
        const normalized = results.map(r => this.normalizeResult(r));

        // 3. 加权融合
        const fused = {
            faults: [],
            confidence: 0,
            severity: 'normal'
        };

        // 合并所有诊断结果
        normalized.forEach((result, index) => {
            const weight = weights[Object.keys(weights)[index]];
            
            result.faults.forEach(fault => {
                const existingFault = fused.faults.find(f => f.name === fault.name);
                if (existingFault) {
                    existingFault.confidence = (existingFault.confidence + fault.confidence * weight) / 2;
                    existingFault.severity = this.fuseSeverity(existingFault.severity, fault.severity);
                } else {
                    fused.faults.push({
                        ...fault,
                        confidence: fault.confidence * weight
                    });
                }
            });
        });

        // 计算总体可信度
        fused.confidence = fused.faults.reduce((acc, f) => acc + f.confidence, 0) / fused.faults.length;

        // 确定最终严重程度
        fused.severity = this.determineOverallSeverity(fused.faults);

        return fused;
    }

    /**
     * 生成诊断报告
     * @private
     */
    generateDiagnosisReport(stationId, portId, result, reliability) {
        return {
            id: `D${Date.now()}`,
            stationId,
            portId,
            timestamp: new Date(),
            faults: result.faults.map(fault => ({
                ...fault,
                suggestedActions: this.generateActionPlan(fault)
            })),
            overallSeverity: result.severity,
            confidence: result.confidence,
            reliability,
            summary: this.generateSummary(result),
            recommendations: this.generateRecommendations(result)
        };
    }

    /**
     * 更新故障历史
     * @private
     */
    updateFaultHistory(stationId, portId, report) {
        const key = `${stationId}-${portId}`;
        if (!this.faultHistory.has(key)) {
            this.faultHistory.set(key, []);
        }
        this.faultHistory.get(key).push(report);

        // 保持历史记录在合理范围内
        const MAX_HISTORY = 100;
        const history = this.faultHistory.get(key);
        if (history.length > MAX_HISTORY) {
            history.splice(0, history.length - MAX_HISTORY);
        }
    }

    /**
     * 生成故障处理建议
     * @private
     */
    generateActionPlan(fault) {
        const actions = [];

        // 1. 紧急措施
        if (fault.severity === 'critical') {
            actions.push({
                type: 'emergency',
                priority: 1,
                action: '立即停止充电并断开连接',
                deadline: 'immediate'
            });
        }

        // 2. 检查步骤
        actions.push(...this.generateCheckSteps(fault));

        // 3. 维修建议
        actions.push(...this.generateRepairSuggestions(fault));

        // 4. 预防措施
        actions.push(...this.generatePreventiveMeasures(fault));

        return actions;
    }

    /**
     * 生成检查步骤
     * @private
     */
    generateCheckSteps(fault) {
        const steps = [];
        
        switch (fault.category) {
            case 'power':
                steps.push(
                    { type: 'check', action: '检查输入电源参数', tools: ['万用表'] },
                    { type: 'check', action: '检查输出电路', tools: ['示波器'] },
                    { type: 'check', action: '测试保护电路', tools: ['测试仪'] }
                );
                break;
            case 'communication':
                steps.push(
                    { type: 'check', action: '检查网络连接状态', tools: ['网络分析仪'] },
                    { type: 'check', action: '测试通信模块', tools: ['协议分析仪'] },
                    { type: 'check', action: '验证配置参数', tools: ['配置工具'] }
                );
                break;
            case 'temperature':
                steps.push(
                    { type: 'check', action: '检查散热系统', tools: ['红外测温仪'] },
                    { type: 'check', action: '测试风扇运行', tools: ['转速表'] },
                    { type: 'check', action: '检查温度传感器', tools: ['温度校准器'] }
                );
                break;
        }

        return steps;
    }

    /**
     * 生成维修建议
     * @private
     */
    generateRepairSuggestions(fault) {
        const suggestions = [];

        // 根据故障类型生成具体维修建议
        switch (fault.category) {
            case 'power':
                suggestions.push(
                    {
                        type: 'repair',
                        action: '更换故障组件',
                        parts: this.getRequiredParts(fault),
                        estimatedTime: '2小时',
                        skillLevel: 'expert'
                    }
                );
                break;
            case 'communication':
                suggestions.push(
                    {
                        type: 'repair',
                        action: '更新通信模块固件',
                        tools: ['编程器'],
                        estimatedTime: '1小时',
                        skillLevel: 'intermediate'
                    }
                );
                break;
            case 'temperature':
                suggestions.push(
                    {
                        type: 'repair',
                        action: '清理散热系统',
                        tools: ['清洁工具套装'],
                        estimatedTime: '1.5小时',
                        skillLevel: 'basic'
                    }
                );
                break;
        }

        return suggestions;
    }

    /**
     * 生成预防措施
     * @private
     */
    generatePreventiveMeasures(fault) {
        const measures = [];

        // 根据故障类型生成预防措施
        switch (fault.category) {
            case 'power':
                measures.push(
                    {
                        type: 'preventive',
                        action: '定期校准电压/电流传感器',
                        interval: '3个月',
                        priority: 'high'
                    },
                    {
                        type: 'preventive',
                        action: '检查电源质量',
                        interval: '1个月',
                        priority: 'medium'
                    }
                );
                break;
            case 'communication':
                measures.push(
                    {
                        type: 'preventive',
                        action: '定期更新通信模块固件',
                        interval: '6个月',
                        priority: 'medium'
                    },
                    {
                        type: 'preventive',
                        action: '监控网络质量',
                        interval: '实时',
                        priority: 'high'
                    }
                );
                break;
            case 'temperature':
                measures.push(
                    {
                        type: 'preventive',
                        action: '定期清理散热系统',
                        interval: '2个月',
                        priority: 'high'
                    },
                    {
                        type: 'preventive',
                        action: '检查风扇运行状态',
                        interval: '1周',
                        priority: 'medium'
                    }
                );
                break;
        }

        return measures;
    }
}

// 导出故障诊断系统
window.FaultDiagnosisSystem = FaultDiagnosisSystem; 