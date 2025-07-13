/**
 * 能源交易执行功能
 */

class EnergyTrading {
    constructor() {
        // 初始化交易参数
        this.tradeType = 'spot';
        this.tradeDirection = 'buy';
        this.powerAmount = 0;
        this.powerPrice = 0;
        this.maxTradeAmount = 1000; // 示例最大可交易量
        
        // 交易类型配置
        this.tradeTypeConfig = {
            spot: {
                name: '现货交易',
                description: '日前/日内电力现货市场交易',
                minAmount: 100,
                maxAmount: 1000,
                priceRange: { min: 300, max: 800 },
                periods: ['peak', 'flat', 'valley'], // 峰时段、平时段、谷时段
                periodConfig: {
                    peak: { name: '峰时段', time: '08:00-21:00', price: 1.2 },
                    flat: { name: '平时段', time: '21:00-24:00', price: 1.0 },
                    valley: { name: '谷时段', time: '00:00-08:00', price: 0.8 }
                },
                validationRules: {
                    priceDeviation: 0.1,
                    minIncrement: 0.01,
                    maxOrdersPerDay: 50
                },
                timeWindows: {
                    dayAhead: { start: '10:00', end: '14:00' },
                    intraday: { start: '08:00', end: '16:00' }
                },
                settlementRules: {
                    type: 'realtime',
                    deadline: '24h'
                }
            },
            contract: {
                name: '合约交易',
                description: '中长期电力合约交易',
                minAmount: 500,
                maxAmount: 5000,
                priceRange: { min: 350, max: 600 },
                periods: ['monthly', 'quarterly', 'yearly'],
                contractTypes: [
                    { id: 'fixed', name: '固定合约', description: '固定电量和价格' },
                    { id: 'flexible', name: '弹性合约', description: '可调节电量' },
                    { id: 'indexed', name: '指数合约', description: '价格与指数挂钩' }
                ],
                validationRules: {
                    minDuration: 30, // 最短合约期限（天）
                    maxDuration: 365, // 最长合约期限（天）
                    creditLimit: 1000000 // 信用额度限制
                },
                performanceBond: {
                    rate: 0.1, // 保证金比例
                    minAmount: 50000 // 最低保证金
                },
                settlementRules: {
                    type: 'monthly', // 月度结算
                    deadline: '5d' // 结算期限
                }
            },
            auxiliary: {
                name: '辅助服务',
                description: '电网辅助服务交易',
                types: [
                    { id: 'frequency', name: '调频服务', minCapacity: 10 },
                    { id: 'reserve', name: '备用服务', minCapacity: 50 },
                    { id: 'voltage', name: '调压服务', minCapacity: 20 },
                    { id: 'blackstart', name: '黑启动', minCapacity: 100 }
                ],
                minAmount: 200,
                maxAmount: 2000,
                priceRange: { min: 200, max: 400 },
                periods: ['hourly', 'daily', 'weekly'],
                responseTime: {
                    frequency: 30, // 调频响应时间（秒）
                    reserve: 300, // 备用响应时间（秒）
                    voltage: 60, // 调压响应时间（秒）
                    blackstart: 1800 // 黑启动响应时间（秒）
                },
                compensationRules: {
                    baseRate: 100, // 基础补偿费率
                    performanceBonus: 0.2, // 性能奖励系数
                    penaltyRate: 0.5 // 考核惩罚系数
                },
                settlementRules: {
                    type: 'weekly', // 周度结算
                    deadline: '3d' // 结算期限
                }
            },
            green: {
                name: '绿色电力交易',
                description: '可再生能源电力交易',
                minAmount: 100,
                maxAmount: 3000,
                priceRange: { min: 400, max: 900 },
                periods: ['monthly', 'quarterly', 'yearly'],
                certificateTypes: [
                    { id: 'wind', name: '风电证书' },
                    { id: 'solar', name: '光伏证书' },
                    { id: 'hydro', name: '水电证书' },
                    { id: 'biomass', name: '生物质证书' }
                ],
                validationRules: {
                    sourceVerification: true, // 来源核验
                    carbonOffset: true // 碳减排核算
                },
                incentives: {
                    subsidyRate: 0.1, // 补贴比例
                    carbonCredit: 0.5 // 碳信用系数
                },
                settlementRules: {
                    type: 'monthly',
                    deadline: '7d'
                }
            }
        };
        
        // 添加交易确认流程配置
        this.confirmationWorkflow = {
            // 交易风险等级定义
            riskLevels: {
                low: { threshold: 100000, approvers: ['trader'] },
                medium: { threshold: 500000, approvers: ['trader', 'supervisor'] },
                high: { threshold: 2000000, approvers: ['trader', 'supervisor', 'risk_manager'] },
                critical: { threshold: Infinity, approvers: ['trader', 'supervisor', 'risk_manager', 'director'] }
            },
            
            // 风险检查项
            riskChecks: {
                creditLimit: {
                    name: '信用额度检查',
                    check: (trade) => this._checkCreditLimit(trade)
                },
                tradingLimit: {
                    name: '交易限额检查',
                    check: (trade) => this._checkTradingLimit(trade)
                },
                priceDeviation: {
                    name: '价格偏离度检查',
                    check: (trade) => this._checkPriceDeviation(trade)
                },
                counterpartyRisk: {
                    name: '交易对手风险检查',
                    check: (trade) => this._checkCounterpartyRisk(trade)
                },
                marketRisk: {
                    name: '市场风险评估',
                    check: (trade) => this._assessMarketRisk(trade)
                },
                regulatoryCompliance: {
                    name: '合规性检查',
                    check: (trade) => this._checkRegulatory(trade)
                }
            },

            // 审批流程配置
            approvalWorkflow: {
                trader: {
                    level: 1,
                    title: '交易员确认',
                    checks: ['basicInfo', 'tradingLimit']
                },
                supervisor: {
                    level: 2,
                    title: '主管审批',
                    checks: ['priceDeviation', 'creditLimit']
                },
                risk_manager: {
                    level: 3,
                    title: '风控经理审批',
                    checks: ['counterpartyRisk', 'marketRisk']
                },
                director: {
                    level: 4,
                    title: '总监审批',
                    checks: ['regulatoryCompliance']
                }
            }
        };
        
        // 等待DOM加载完成后初始化
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }
    
    /**
     * 初始化所有功能
     */
    initialize() {
        this.initializeCharts();
        this.initializeEventListeners();
        this.updateCurrentPrice();
        this.updateTradeTypeUI(this.tradeType);
    }
    
    /**
     * 初始化所有事件监听器
     */
    initializeEventListeners() {
        // 交易类型切换
        document.querySelectorAll('input[name="trading-type"]').forEach(radio => {
            radio.addEventListener('change', (e) => this.handleTradeTypeChange(e));
        });
        
        // 交易方向切换
        const buyButton = document.getElementById('buyButton');
        const sellButton = document.getElementById('sellButton');
        
        if (buyButton && sellButton) {
            buyButton.addEventListener('click', () => this.handleDirectionChange('buy'));
            sellButton.addEventListener('click', () => this.handleDirectionChange('sell'));
        }
        
        // 电量快捷按钮事件
        document.querySelectorAll('[data-amount]').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleQuickAmount(e));
        });
        
        // 监听输入变化
        const powerAmountInput = document.getElementById('powerAmount');
        const powerPriceInput = document.getElementById('powerPrice');
        
        if (powerAmountInput) {
            powerAmountInput.addEventListener('input', (e) => this.handleAmountInput(e));
        }
        
        if (powerPriceInput) {
            powerPriceInput.addEventListener('input', (e) => this.handlePriceInput(e));
        }
        
        // 交易时段变更
        const tradingPeriodSelect = document.getElementById('tradingPeriod');
        if (tradingPeriodSelect) {
            tradingPeriodSelect.addEventListener('change', () => {
                this.handleTradingPeriodChange();
                this.updateEstimatedAmount();
            });
        }
        
        // 确认交易按钮
        const tradingForm = document.getElementById('tradingForm');
        if (tradingForm) {
            tradingForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleTradeConfirm();
            });
        }
        
        // 执行交易按钮
        const executeTradeBtn = document.getElementById('executeTradeBtn');
        if (executeTradeBtn) {
            executeTradeBtn.addEventListener('click', () => this.handleTradeExecution());
        }
        
        // 图表周期切换按钮
        document.querySelectorAll('.chart-controls .btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.handlePeriodChange(e));
        });
    }
    
    /**
     * 初始化交易图表
     */
    initializeCharts() {
        // 创建图表实例
        this.charts = new TradingCharts();
        
        // 将更新周期方法绑定到window对象，供其他模块调用
        window.updateChartPeriod = (period) => {
            this.charts.updatePeriod(period);
        };
    }
    
    /**
     * 更新当前市场价格
     */
    updateCurrentPrice() {
        const currentPriceElement = document.getElementById('currentMarketPrice');
        if (currentPriceElement) {
            // 获取最新的K线数据点的收盘价
            const latestData = this.charts.chartData[this.charts.chartData.length - 1];
            const currentPrice = latestData.y[3]; // 收盘价
            currentPriceElement.textContent = currentPrice.toFixed(2);
            
            // 设置为默认交易价格
            const powerPriceInput = document.getElementById('powerPrice');
            if (powerPriceInput) {
                powerPriceInput.value = currentPrice.toFixed(2);
                this.powerPrice = currentPrice;
                this.updateEstimatedAmount();
            }
        }
    }
    
    /**
     * 处理交易类型变更
     */
    handleTradeTypeChange(event) {
        const newType = event.target.value;
        this.tradeType = newType;
        this.updateTradeTypeUI(newType);
        this.updateTradeForm(newType);
    }
    
    /**
     * 更新交易类型相关UI
     */
    updateTradeTypeUI(type) {
        const config = this.tradeTypeConfig[type];
        if (!config) return;
        
        // 更新最大可交易量显示
        const maxAmountElement = document.getElementById('maxAmount');
        if (maxAmountElement) {
            maxAmountElement.textContent = `最大可交易量: ${config.maxAmount} MWh`;
            this.maxTradeAmount = config.maxAmount;
        }
        
        // 更新电量输入限制
        const powerAmountInput = document.getElementById('powerAmount');
        if (powerAmountInput) {
            powerAmountInput.min = config.minAmount;
            powerAmountInput.max = config.maxAmount;
            powerAmountInput.placeholder = `输入交易电量 (${config.minAmount}-${config.maxAmount} MWh)`;
        }
        
        // 更新价格输入限制
        const powerPriceInput = document.getElementById('powerPrice');
        if (powerPriceInput) {
            powerPriceInput.min = config.priceRange.min;
            powerPriceInput.max = config.priceRange.max;
            powerPriceInput.placeholder = `输入交易价格 (${config.priceRange.min}-${config.priceRange.max} 元/MWh)`;
        }
        
        // 更新交易时段选项
        const tradingPeriodSelect = document.getElementById('tradingPeriod');
        if (tradingPeriodSelect) {
            // 清空现有选项
            tradingPeriodSelect.innerHTML = '';
            
            // 根据交易类型添加相应的时段选项
            if (type === 'spot') {
                // 现货交易使用峰平谷时段
                Object.entries(config.periodConfig).forEach(([value, data]) => {
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = `${data.name} (${data.time})`;
                    tradingPeriodSelect.appendChild(option);
                });
            } else {
                // 其他类型使用配置中定义的时段
                config.periods.forEach(period => {
                    const option = document.createElement('option');
                    option.value = period;
                    option.textContent = this.formatPeriod(period);
                    tradingPeriodSelect.appendChild(option);
                });
            }
            
            // 选择第一个选项
            if (tradingPeriodSelect.options.length > 0) {
                tradingPeriodSelect.selectedIndex = 0;
            }
        }
        
        // 重置表单数据
        this.resetForm();
    }
    
    /**
     * 更新交易表单
     */
    updateTradeForm(type) {
        const config = this.tradeTypeConfig[type];
        if (!config) return;

        const formContainer = document.querySelector('.trading-form');
        if (!formContainer) return;

        // 清除原有的特定类型字段
        const specificFields = formContainer.querySelector('.type-specific-fields');
        if (specificFields) {
            specificFields.remove();
        }

        // 创建新的特定类型字段
        const newFields = document.createElement('div');
        newFields.className = 'type-specific-fields';

        switch (type) {
            case 'contract':
                newFields.innerHTML = this.generateContractFields(config);
                break;
            case 'auxiliary':
                newFields.innerHTML = this.generateAuxiliaryFields(config);
                break;
            case 'green':
                newFields.innerHTML = this.generateGreenFields(config);
                break;
            default:
                newFields.innerHTML = this.generateSpotFields(config);
        }

        // 插入新字段
        const submitButton = formContainer.querySelector('button[type="submit"]');
        formContainer.insertBefore(newFields, submitButton);

        // 绑定新字段的事件监听器
        this.bindTypeSpecificEvents(type);
    }

    /**
     * 生成合约交易特定字段
     */
    generateContractFields(config) {
        return `
            <div class="mb-3">
                <label class="form-label">合约类型</label>
                <select class="form-select" id="contractType">
                    ${config.contractTypes.map(type => 
                        `<option value="${type.id}">${type.name} - ${type.description}</option>`
                    ).join('')}
                </select>
            </div>
            <div class="mb-3">
                <label class="form-label">合约期限</label>
                <div class="input-group">
                    <input type="number" class="form-control" id="contractDuration" 
                           min="${config.validationRules.minDuration}" 
                           max="${config.validationRules.maxDuration}">
                    <span class="input-group-text">天</span>
                </div>
            </div>
            <div class="mb-3">
                <label class="form-label">履约保证金</label>
                <div class="input-group">
                    <span class="input-group-text">¥</span>
                    <input type="number" class="form-control" id="performanceBond" 
                           min="${config.performanceBond.minAmount}">
                </div>
                <small class="text-muted">最低保证金：¥${config.performanceBond.minAmount}</small>
            </div>
        `;
    }

    /**
     * 生成辅助服务特定字段
     */
    generateAuxiliaryFields(config) {
        return `
            <div class="mb-3">
                <label class="form-label">服务类型</label>
                <select class="form-select" id="serviceType">
                    ${config.types.map(type => 
                        `<option value="${type.id}">${type.name} (最小容量: ${type.minCapacity}MW)</option>`
                    ).join('')}
                </select>
            </div>
            <div class="mb-3">
                <label class="form-label">响应时间要求</label>
                <div class="input-group">
                    <input type="number" class="form-control" id="responseTime" readonly>
                    <span class="input-group-text">秒</span>
                </div>
            </div>
            <div class="mb-3">
                <label class="form-label">服务时段</label>
                <select class="form-select" id="servicePeriod">
                    ${config.periods.map(period => 
                        `<option value="${period}">${this.formatPeriod(period)}</option>`
                    ).join('')}
                </select>
            </div>
        `;
    }

    /**
     * 生成绿色电力交易特定字段
     */
    generateGreenFields(config) {
        return `
            <div class="mb-3">
                <label class="form-label">证书类型</label>
                <select class="form-select" id="certificateType">
                    ${config.certificateTypes.map(cert => 
                        `<option value="${cert.id}">${cert.name}</option>`
                    ).join('')}
                </select>
            </div>
            <div class="mb-3">
                <label class="form-label">交易期限</label>
                <select class="form-select" id="greenPeriod">
                    ${config.periods.map(period => 
                        `<option value="${period}">${this.formatPeriod(period)}</option>`
                    ).join('')}
                </select>
            </div>
            <div class="mb-3">
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="sourceVerification">
                    <label class="form-check-label">需要来源核验</label>
                </div>
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="carbonOffset">
                    <label class="form-check-label">包含碳减排核算</label>
                </div>
            </div>
        `;
    }

    /**
     * 生成现货交易特定字段
     */
    generateSpotFields(config) {
        return `
            <div class="mb-3">
                <label class="form-label">交易时段</label>
                <select class="form-select" id="spotPeriod">
                    <option value="dayAhead">日前市场 (${config.timeWindows.dayAhead.start}-${config.timeWindows.dayAhead.end})</option>
                    <option value="intraday">日内市场 (${config.timeWindows.intraday.start}-${config.timeWindows.intraday.end})</option>
                </select>
            </div>
            <div class="mb-3">
                <label class="form-label">出清方式</label>
                <select class="form-select" id="clearingMethod">
                    <option value="continuous">连续竞价</option>
                    <option value="call">集中竞价</option>
                </select>
            </div>
        `;
    }

    /**
     * 绑定特定类型的事件监听器
     */
    bindTypeSpecificEvents(type) {
        switch (type) {
            case 'contract':
                this.bindContractEvents();
                break;
            case 'auxiliary':
                this.bindAuxiliaryEvents();
                break;
            case 'green':
                this.bindGreenEvents();
                break;
            default:
                this.bindSpotEvents();
        }
    }

    /**
     * 格式化时间段显示
     */
    formatPeriod(period) {
        const periodMap = {
            hourly: '小时',
            daily: '日',
            weekly: '周',
            monthly: '月',
            quarterly: '季度',
            yearly: '年'
        };
        return periodMap[period] || period;
    }
    
    /**
     * 处理交易方向变更
     */
    handleDirectionChange(direction) {
        this.tradeDirection = direction;
        
        // 更新按钮状态
        const buyButton = document.getElementById('buyButton');
        const sellButton = document.getElementById('sellButton');
        
        if (buyButton && sellButton) {
            if (direction === 'buy') {
                buyButton.classList.remove('btn-outline-success');
                buyButton.classList.add('btn-success');
                sellButton.classList.remove('btn-danger');
                sellButton.classList.add('btn-outline-danger');
            } else {
                buyButton.classList.remove('btn-success');
                buyButton.classList.add('btn-outline-success');
                sellButton.classList.remove('btn-outline-danger');
                sellButton.classList.add('btn-danger');
            }
        }
    }
    
    /**
     * 处理快捷电量按钮点击
     */
    handleQuickAmount(event) {
        const percentage = parseInt(event.target.dataset.amount);
        this.powerAmount = (this.maxTradeAmount * percentage / 100);
        document.getElementById('powerAmount').value = this.powerAmount;
        this.updateEstimatedAmount();
    }
    
    /**
     * 处理电量输入
     */
    handleAmountInput(event) {
        this.powerAmount = parseFloat(event.target.value) || 0;
        this.updateEstimatedAmount();
    }
    
    /**
     * 处理价格输入
     */
    handlePriceInput(event) {
        this.powerPrice = parseFloat(event.target.value) || 0;
        this.updateEstimatedAmount();
    }
    
    /**
     * 更新估算金额
     */
    updateEstimatedAmount() {
        const amount = this.powerAmount * this.powerPrice;
        const estimatedAmountElement = document.getElementById('estimatedAmount');
        if (estimatedAmountElement) {
            estimatedAmountElement.textContent = amount.toFixed(2);
        }
    }
    
    /**
     * 处理交易执行
     */
    handleTradeExecution() {
        // 这里添加实际的交易执行逻辑
        this.showToast('success', '交易已提交成功！');
        
        // 关闭确认弹窗
        const modal = document.getElementById('tradeConfirmModal');
        if (modal) {
            bootstrap.Modal.getInstance(modal).hide();
        }
        
        // 重置表单
        this.resetForm();
    }
    
    /**
     * 重置表单
     */
    resetForm() {
        const powerAmountInput = document.getElementById('powerAmount');
        const powerPriceInput = document.getElementById('powerPrice');
        
        if (powerAmountInput) {
            powerAmountInput.value = '';
        }
        if (powerPriceInput) {
            powerPriceInput.value = '';
        }
        
        this.powerAmount = 0;
        this.powerPrice = 0;
        this.updateEstimatedAmount();
    }
    
    /**
     * 处理图表周期变更
     */
    handlePeriodChange(event) {
        const period = event.target.dataset.period;
        document.querySelectorAll('.chart-controls .btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');
        
        if (this.charts) {
            this.charts.updatePeriod(period);
        }
    }
    
    /**
     * 显示Toast提示
     */
    showToast(type, message) {
        const toast = document.createElement('div');
        toast.className = `toast bg-${type} text-white`;
        toast.innerHTML = `
            <div class="toast-body">
                ${message}
            </div>
        `;
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // 自动移除
        toast.addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toast);
        });
    }
    
    /**
     * 验证交易输入
     */
    validateTradeInput() {
        const config = this.tradeTypeConfig[this.tradeType];
        if (!config) return false;
        
        const errors = [];
        
        // 验证电量
        if (!this.powerAmount) {
            errors.push('请输入交易电量');
        } else if (this.powerAmount < config.minAmount || this.powerAmount > config.maxAmount) {
            errors.push(`交易电量必须在 ${config.minAmount}-${config.maxAmount} MWh 之间`);
        }
        
        // 验证价格
        if (!this.powerPrice) {
            errors.push('请输入交易价格');
        } else if (this.powerPrice < config.priceRange.min || this.powerPrice > config.priceRange.max) {
            errors.push(`交易价格必须在 ${config.priceRange.min}-${config.priceRange.max} 元/MWh 之间`);
        }
        
        // 验证交易时段
        const tradingPeriodSelect = document.getElementById('tradingPeriod');
        if (tradingPeriodSelect && !config.periods.includes(tradingPeriodSelect.value)) {
            errors.push('当前交易类型不支持所选交易时段');
        }
        
        // 特定类型验证
        const typeSpecificErrors = this.validateTypeSpecific();
        errors.push(...typeSpecificErrors);
        
        if (errors.length > 0) {
            this.showToast('warning', errors.join('\n'));
            return false;
        }
        
        return true;
    }
    
    /**
     * 特定类型验证
     */
    validateTypeSpecific() {
        const errors = [];
        const config = this.tradeTypeConfig[this.tradeType];

        switch (this.tradeType) {
            case 'contract':
                const duration = document.getElementById('contractDuration')?.value;
                const bond = document.getElementById('performanceBond')?.value;

                if (duration < config.validationRules.minDuration || 
                    duration > config.validationRules.maxDuration) {
                    errors.push(`合约期限必须在 ${config.validationRules.minDuration}-${config.validationRules.maxDuration} 天之间`);
                }

                if (bond < config.performanceBond.minAmount) {
                    errors.push(`履约保证金不能低于 ¥${config.performanceBond.minAmount}`);
                }
                break;

            case 'auxiliary':
                const serviceType = document.getElementById('serviceType')?.value;
                const typeConfig = config.types.find(t => t.id === serviceType);
                
                if (this.powerAmount < typeConfig?.minCapacity) {
                    errors.push(`${typeConfig.name}最小容量要求为 ${typeConfig.minCapacity}MW`);
                }
                break;

            case 'green':
                const sourceVerification = document.getElementById('sourceVerification')?.checked;
                const carbonOffset = document.getElementById('carbonOffset')?.checked;

                if (config.validationRules.sourceVerification && !sourceVerification) {
                    errors.push('绿色电力交易必须进行来源核验');
                }
                break;
        }

        return errors;
    }
    
    /**
     * 处理交易确认
     */
    async handleTradeConfirm() {
        if (!this.validateTradeInput()) {
            return;
        }

        // 构建交易对象
        const trade = this._buildTradeObject();
        
        // 执行风险评估
        const riskAssessment = await this._performRiskAssessment(trade);
        
        // 确定风险等级和所需审批流程
        const { riskLevel, requiredApprovals } = this._determineRiskLevel(trade, riskAssessment);
        
        // 显示风险评估结果
        this._showRiskAssessmentModal(trade, riskAssessment, riskLevel, requiredApprovals);
    }
    
    /**
     * 构建交易对象
     */
    _buildTradeObject() {
        return {
            id: `TRD${Date.now()}`,
            type: this.tradeType,
            direction: this.tradeDirection,
            amount: this.powerAmount,
            price: this.powerPrice,
            totalValue: this.powerAmount * this.powerPrice,
            period: document.getElementById('tradingPeriod')?.value,
            timestamp: new Date(),
            additionalInfo: this._getAdditionalTradeInfo()
        };
    }

    /**
     * 获取额外的交易信息
     */
    _getAdditionalTradeInfo() {
        const info = {};
        
        switch (this.tradeType) {
            case 'contract':
                info.contractType = document.getElementById('contractType')?.value;
                info.duration = document.getElementById('contractDuration')?.value;
                info.performanceBond = document.getElementById('performanceBond')?.value;
                break;
            case 'auxiliary':
                info.serviceType = document.getElementById('serviceType')?.value;
                info.responseTime = document.getElementById('responseTime')?.value;
                break;
            case 'green':
                info.certificateType = document.getElementById('certificateType')?.value;
                info.sourceVerification = document.getElementById('sourceVerification')?.checked;
                info.carbonOffset = document.getElementById('carbonOffset')?.checked;
                break;
        }
        
        return info;
    }

    /**
     * 执行风险评估
     */
    async _performRiskAssessment(trade) {
        const assessment = {
            checks: {},
            warnings: [],
            blockers: []
        };

        // 执行所有风险检查
        for (const [key, check] of Object.entries(this.confirmationWorkflow.riskChecks)) {
            const result = await check.check(trade);
            assessment.checks[key] = result;
            
            if (result.status === 'warning') {
                assessment.warnings.push(result);
            } else if (result.status === 'blocked') {
                assessment.blockers.push(result);
            }
        }

        return assessment;
    }

    /**
     * 确定风险等级和所需审批
     */
    _determineRiskLevel(trade, assessment) {
        // 基于交易金额确定初始风险等级
        let riskLevel = 'low';
        for (const [level, config] of Object.entries(this.confirmationWorkflow.riskLevels)) {
            if (trade.totalValue <= config.threshold) {
                riskLevel = level;
                break;
            }
        }

        // 根据风险评估结果可能提升风险等级
        if (assessment.blockers.length > 0) {
            riskLevel = 'critical';
        } else if (assessment.warnings.length >= 3) {
            riskLevel = 'high';
        } else if (assessment.warnings.length > 0) {
            // 提升一个风险等级
            const levels = ['low', 'medium', 'high', 'critical'];
            const currentIndex = levels.indexOf(riskLevel);
            if (currentIndex < levels.length - 1) {
                riskLevel = levels[currentIndex + 1];
            }
        }

        return {
            riskLevel,
            requiredApprovals: this.confirmationWorkflow.riskLevels[riskLevel].approvers
        };
    }

    /**
     * 显示风险评估模态框
     */
    _showRiskAssessmentModal(trade, assessment, riskLevel, requiredApprovals) {
        // 创建模态框内容
        const modalContent = this._createRiskAssessmentModalContent(trade, assessment, riskLevel, requiredApprovals);
        
        // 更新或创建模态框
        let modal = document.getElementById('riskAssessmentModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'riskAssessmentModal';
            modal.className = 'modal fade';
            document.body.appendChild(modal);
        }
        
        modal.innerHTML = modalContent;
        
        // 显示模态框
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        // 绑定事件处理
        this._bindRiskAssessmentEvents(modal, trade, assessment, requiredApprovals);
    }

    /**
     * 创建风险评估模态框内容
     */
    _createRiskAssessmentModalContent(trade, assessment, riskLevel, requiredApprovals) {
        const riskLevelClass = {
            low: 'success',
            medium: 'warning',
            high: 'danger',
            critical: 'dark'
        }[riskLevel];

        return `
            <div class="modal-dialog modal-lg modal-dialog-scrollable">
                <div class="modal-content">
                    <div class="modal-header bg-${riskLevelClass} text-white">
                        <h5 class="modal-title">交易风险评估 - ${this.tradeTypeConfig[trade.type].name}</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <!-- 交易基本信息 -->
                        <div class="card mb-3">
                            <div class="card-header">
                                <h6 class="mb-0">交易信息</h6>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <p><strong>交易ID：</strong> ${trade.id}</p>
                                        <p><strong>交易方向：</strong> ${trade.direction === 'buy' ? '买入' : '卖出'}</p>
                                        <p><strong>交易电量：</strong> ${trade.amount} MWh</p>
                                    </div>
                                    <div class="col-md-6">
                                        <p><strong>交易价格：</strong> ¥${trade.price}/MWh</p>
                                        <p><strong>交易总额：</strong> ¥${trade.totalValue.toFixed(2)}</p>
                                        <p><strong>风险等级：</strong> <span class="badge bg-${riskLevelClass}">${riskLevel.toUpperCase()}</span></p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 风险评估结果 -->
                        <div class="card mb-3">
                            <div class="card-header">
                                <h6 class="mb-0">风险评估结果</h6>
                            </div>
                            <div class="card-body p-0">
                                <div class="table-responsive">
                                    <table class="table mb-0">
                                        <thead>
                                            <tr>
                                                <th>检查项</th>
                                                <th>状态</th>
                                                <th>详情</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${Object.entries(assessment.checks).map(([key, result]) => `
                                                <tr>
                                                    <td>${this.confirmationWorkflow.riskChecks[key].name}</td>
                                                    <td>
                                                        <span class="badge bg-${
                                                            result.status === 'passed' ? 'success' : 
                                                            result.status === 'warning' ? 'warning' : 'danger'
                                                        }">
                                                            ${result.status.toUpperCase()}
                                                        </span>
                                                    </td>
                                                    <td>${result.message}</td>
                                                </tr>
                                            `).join('')}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>

                        <!-- 审批流程 -->
                        <div class="card">
                            <div class="card-header">
                                <h6 class="mb-0">审批流程</h6>
                            </div>
                            <div class="card-body">
                                <div class="approval-workflow">
                                    ${requiredApprovals.map((role, index) => `
                                        <div class="approval-step ${index === 0 ? 'current' : ''}">
                                            <div class="step-number">${index + 1}</div>
                                            <div class="step-content">
                                                <h6>${this.confirmationWorkflow.approvalWorkflow[role].title}</h6>
                                                <p class="mb-0 text-muted">
                                                    检查项：${this.confirmationWorkflow.approvalWorkflow[role].checks.join(', ')}
                                                </p>
                                            </div>
                                            <div class="step-status">
                                                <span class="badge bg-secondary">待审批</span>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>

                        ${assessment.blockers.length > 0 ? `
                            <div class="alert alert-danger mt-3">
                                <h6 class="alert-heading">交易被阻止</h6>
                                <ul class="mb-0">
                                    ${assessment.blockers.map(blocker => `
                                        <li>${blocker.message}</li>
                                    `).join('')}
                                </ul>
                            </div>
                        ` : ''}

                        ${assessment.warnings.length > 0 ? `
                            <div class="alert alert-warning mt-3">
                                <h6 class="alert-heading">风险提示</h6>
                                <ul class="mb-0">
                                    ${assessment.warnings.map(warning => `
                                        <li>${warning.message}</li>
                                    `).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                        ${assessment.blockers.length === 0 ? `
                            <button type="button" class="btn btn-primary" id="startApprovalBtn">
                                启动审批流程
                            </button>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * 绑定风险评估模态框事件
     */
    _bindRiskAssessmentEvents(modal, trade, assessment, requiredApprovals) {
        const startApprovalBtn = modal.querySelector('#startApprovalBtn');
        if (startApprovalBtn) {
            startApprovalBtn.addEventListener('click', () => {
                this._startApprovalWorkflow(trade, assessment, requiredApprovals);
            });
        }
    }

    /**
     * 启动审批流程
     */
    _startApprovalWorkflow(trade, assessment, requiredApprovals) {
        // 创建审批任务
        const approvalTask = {
            id: `APV${Date.now()}`,
            tradeId: trade.id,
            trade: trade,
            assessment: assessment,
            requiredApprovals: requiredApprovals,
            currentApprovalIndex: 0,
            approvalHistory: [],
            status: 'pending'
        };

        // 保存审批任务
        this._saveApprovalTask(approvalTask);

        // 通知第一个审批人
        this._notifyNextApprover(approvalTask);

        // 关闭风险评估模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('riskAssessmentModal'));
        modal.hide();

        // 显示提示
        this.showToast('success', '审批流程已启动，请等待审批');
    }

    /**
     * 保存审批任务
     */
    _saveApprovalTask(task) {
        // 这里应该调用后端API保存审批任务
        console.log('保存审批任务:', task);
    }

    /**
     * 通知下一个审批人
     */
    _notifyNextApprover(task) {
        const currentApprover = task.requiredApprovals[task.currentApprovalIndex];
        const approverConfig = this.confirmationWorkflow.approvalWorkflow[currentApprover];

        // 这里应该调用通知系统API
        console.log(`通知 ${approverConfig.title} 进行审批`);
    }

    /**
     * 信用额度检查
     */
    async _checkCreditLimit(trade) {
        // 这里应该调用实际的信用额度检查API
        const creditLimit = 1000000; // 示例信用额度
        const usedCredit = 500000;   // 示例已用额度
        
        if (trade.totalValue + usedCredit > creditLimit) {
            return {
                status: 'blocked',
                message: `交易金额超过可用信用额度 (可用额度: ¥${(creditLimit - usedCredit).toFixed(2)})`
            };
        }
        
        if (trade.totalValue + usedCredit > creditLimit * 0.8) {
            return {
                status: 'warning',
                message: `交易后信用额度使用率将超过80%`
            };
        }
        
        return {
            status: 'passed',
            message: '信用额度检查通过'
        };
    }

    /**
     * 交易限额检查
     */
    async _checkTradingLimit(trade) {
        const config = this.tradeTypeConfig[trade.type];
        
        if (trade.amount > config.maxAmount) {
            return {
                status: 'blocked',
                message: `交易电量超过最大限制 (${config.maxAmount} MWh)`
            };
        }
        
        if (trade.amount > config.maxAmount * 0.8) {
            return {
                status: 'warning',
                message: `交易电量接近最大限制`
            };
        }
        
        return {
            status: 'passed',
            message: '交易限额检查通过'
        };
    }

    /**
     * 价格偏离度检查
     */
    async _checkPriceDeviation(trade) {
        const currentPrice = parseFloat(document.getElementById('currentMarketPrice')?.textContent || '0');
        const deviation = Math.abs(trade.price - currentPrice) / currentPrice;
        
        if (deviation > 0.1) {
            return {
                status: 'blocked',
                message: `价格偏离度超过10% (当前市场价: ¥${currentPrice})`
            };
        }
        
        if (deviation > 0.05) {
            return {
                status: 'warning',
                message: `价格偏离度较大，请确认`
            };
        }
        
        return {
            status: 'passed',
            message: '价格偏离度在合理范围内'
        };
    }

    /**
     * 交易对手风险检查
     */
    async _checkCounterpartyRisk(trade) {
        // 这里应该调用实际的交易对手风险评估API
        return {
            status: 'passed',
            message: '交易对手风险评估通过'
        };
    }

    /**
     * 市场风险评估
     */
    async _assessMarketRisk(trade) {
        // 这里应该调用实际的市场风险评估API
        const volatility = 0.05; // 示例市场波动率
        
        if (volatility > 0.1) {
            return {
                status: 'warning',
                message: '当前市场波动较大，建议谨慎交易'
            };
        }
        
        return {
            status: 'passed',
            message: '市场风险在可接受范围内'
        };
    }

    /**
     * 合规性检查
     */
    async _checkRegulatory(trade) {
        // 这里应该调用实际的合规检查API
        return {
            status: 'passed',
            message: '合规性检查通过'
        };
    }
}

// 创建交易功能实例
window.energyTrading = new EnergyTrading(); 