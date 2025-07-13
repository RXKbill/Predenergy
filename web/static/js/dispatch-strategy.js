// 调度策略管理类
class DispatchStrategy {
    constructor() {
        this.currentStrategy = null;
        this.dispatchLogs = [];
        this.stations = new Map();
        this.initialize();
    }

    // 初始化
    initialize() {
        this.loadDefaultStrategy();
        this.bindEvents();
        this.startAutoRefresh();
    }

    // 加载默认策略
    loadDefaultStrategy() {
        // 默认策略配置
        this.currentStrategy = {
            runningMode: 'auto',
            loadBalance: 'smart',
            peakTime: {
                enabled: true,
                start: '08:00',
                end: '20:00'
            },
            powerLimits: {
                max: 120,
                min: 30
            },
            timing: {
                response: 3,
                buffer: 5
            },
            priorities: {
                fastCharging: {
                    level: 1,
                    power: 60
                },
                slowCharging: {
                    level: 2,
                    power: 40
                }
            }
        };

        // 更新UI
        this.updateStrategyUI();
        
        // 添加初始日志
        this.addLog('系统初始化完成，加载默认策略');
    }

    // 绑定事件
    bindEvents() {
        // 运行模式切换
        document.getElementById('runningMode').addEventListener('change', (e) => {
            this.currentStrategy.runningMode = e.target.value;
            this.addLog(`切换运行模式: ${this.getModeName(e.target.value)}`);
            this.applyStrategy();
        });

        // 负载均衡方式切换
        document.getElementById('loadBalance').addEventListener('change', (e) => {
            this.currentStrategy.loadBalance = e.target.value;
            this.addLog(`切换负载均衡方式: ${this.getBalanceName(e.target.value)}`);
            this.applyStrategy();
        });

        // 高峰时段启用/禁用
        document.getElementById('peakTimeEnabled').addEventListener('change', (e) => {
            this.currentStrategy.peakTime.enabled = e.target.checked;
            this.addLog(`${e.target.checked ? '启用' : '禁用'}高峰时段管理`);
            this.applyStrategy();
        });

        // 保存策略按钮
        document.querySelector('[onclick="saveStrategy()"]').addEventListener('click', () => this.saveStrategy());

        // 应用策略按钮
        document.querySelector('[onclick="applyStrategy()"]').addEventListener('click', () => this.applyStrategy());

        // 清空日志按钮
        document.querySelector('[onclick="clearLog()"]').addEventListener('click', () => this.clearLogs());

        // 功率调节按钮
        document.querySelector('[onclick="adjustPower()"]').addEventListener('click', () => this.adjustPower());

        // 价格调整按钮
        document.querySelector('[onclick="adjustPrice()"]').addEventListener('click', () => this.adjustPrice());

        // 限流控制按钮
        document.querySelector('[onclick="limitCurrent()"]').addEventListener('click', () => this.limitCurrent());

        // 模式切换下拉框
        document.getElementById('operationMode').addEventListener('change', (e) => this.switchMode(e.target.value));

        // 批量调度按钮
        document.querySelector('[onclick="batchDispatch()"]').addEventListener('click', () => this.batchDispatch());

        // 负载优化按钮
        document.querySelector('[onclick="optimizeLoad()"]').addEventListener('click', () => this.optimizeLoad());

        // 应急模式按钮
        document.querySelector('[onclick="emergencyMode()"]').addEventListener('click', () => this.emergencyMode());
    }

    // 保存策略
    saveStrategy() {
        // 收集表单数据
        this.currentStrategy = {
            ...this.currentStrategy,
            powerLimits: {
                max: parseFloat(document.getElementById('maxPowerLimit').value),
                min: parseFloat(document.getElementById('minPowerGuarantee').value)
            },
            timing: {
                response: parseFloat(document.getElementById('responseTime').value),
                buffer: parseFloat(document.getElementById('bufferTime').value)
            },
            peakTime: {
                enabled: document.getElementById('peakTimeEnabled').checked,
                start: document.getElementById('peakTimeStart').value,
                end: document.getElementById('peakTimeEnd').value
            }
        };

        this.addLog('策略配置已保存');
        this.showToast('策略保存成功', 'success');
    }

    // 应用策略
    applyStrategy() {
        // 模拟策略应用过程
        this.addLog('开始应用新策略...');
        
        setTimeout(() => {
            // 模拟策略应用结果
            const success = Math.random() > 0.1; // 90%成功率
            if (success) {
                this.addLog('策略应用成功，系统已按新配置运行');
                this.showToast('策略应用成功', 'success');
                
                // 模拟一些实时调整
                this.simulateRealTimeAdjustments();
            } else {
                this.addLog('策略应用失败，已回滚到之前配置', 'error');
                this.showToast('策略应用失败', 'danger');
            }
        }, 1500);
    }

    // 模拟实时调整
    simulateRealTimeAdjustments() {
        const adjustments = [
            {
                time: 2000,
                action: () => {
                    this.addLog('检测到负载变化，自动调整分配比例');
                    document.querySelector('input[value="60"]').value = "65";
                    document.querySelector('input[value="40"]').value = "35";
                }
            },
            {
                time: 4000,
                action: () => {
                    this.addLog('响应峰谷电价，调整充电功率限制');
                    document.getElementById('maxPowerLimit').value = "110";
                }
            },
            {
                time: 6000,
                action: () => {
                    this.addLog('优化完成，系统运行稳定');
                }
            }
        ];

        adjustments.forEach(adj => {
            setTimeout(adj.action, adj.time);
        });
    }

    // 添加日志
    addLog(message, type = 'info') {
        const now = new Date();
        const time = now.toLocaleTimeString();
        const log = { time, message, type };
        this.dispatchLogs.unshift(log);

        // 更新UI
        const logContainer = document.querySelector('.log-container');
        const logElement = document.createElement('div');
        logElement.className = 'log-item';
        logElement.innerHTML = `
            <small class="text-muted">${time}</small>
            <span class="text-${this.getLogTypeClass(type)}">${message}</span>
        `;
        logContainer.insertBefore(logElement, logContainer.firstChild);

        // 限制日志数量
        if (this.dispatchLogs.length > 50) {
            this.dispatchLogs.pop();
            if (logContainer.lastChild) {
                logContainer.removeChild(logContainer.lastChild);
            }
        }
    }

    // 清空日志
    clearLogs() {
        this.dispatchLogs = [];
        document.querySelector('.log-container').innerHTML = '';
        this.addLog('日志已清空');
    }

    // 自动刷新
    startAutoRefresh() {
        setInterval(() => {
            // 模拟实时数据更新
            if (this.currentStrategy.runningMode === 'auto') {
                const events = [
                    '负载均衡自动调整完成',
                    '检测到充电高峰，启动智能分配',
                    '系统性能优化完成',
                    '响应时间优化完成'
                ];
                if (Math.random() < 0.3) { // 30%概率生成日志
                    this.addLog(events[Math.floor(Math.random() * events.length)]);
                }
            }
        }, 10000); // 每10秒检查一次
    }

    // 更新策略UI
    updateStrategyUI() {
        // 更新运行模式
        document.getElementById('runningMode').value = this.currentStrategy.runningMode;
        document.getElementById('loadBalance').value = this.currentStrategy.loadBalance;
        
        // 更新高峰时段
        document.getElementById('peakTimeEnabled').checked = this.currentStrategy.peakTime.enabled;
        document.getElementById('peakTimeStart').value = this.currentStrategy.peakTime.start;
        document.getElementById('peakTimeEnd').value = this.currentStrategy.peakTime.end;
        
        // 更新功率限制
        document.getElementById('maxPowerLimit').value = this.currentStrategy.powerLimits.max;
        document.getElementById('minPowerGuarantee').value = this.currentStrategy.powerLimits.min;
        
        // 更新时间设置
        document.getElementById('responseTime').value = this.currentStrategy.timing.response;
        document.getElementById('bufferTime').value = this.currentStrategy.timing.buffer;
    }

    // 获取模式名称
    getModeName(mode) {
        const modes = {
            auto: '自动调度',
            manual: '手动控制',
            eco: '节能模式',
            peak: '高峰应对'
        };
        return modes[mode] || mode;
    }

    // 获取均衡方式名称
    getBalanceName(balance) {
        const balances = {
            smart: '智能分配',
            even: '均匀分配',
            priority: '优先级分配'
        };
        return balances[balance] || balance;
    }

    // 获取日志类型样式
    getLogTypeClass(type) {
        const types = {
            info: 'info',
            success: 'success',
            warning: 'warning',
            error: 'danger'
        };
        return types[type] || 'info';
    }

    // 显示提示消息
    showToast(message, type) {
        const toast = document.getElementById('recordCancelToast');
        const toastBody = document.getElementById('recordCancelToastBody');
        
        toastBody.textContent = message;
        toast.classList.remove('bg-success', 'bg-danger', 'bg-warning');
        toast.classList.add(`bg-${type}`);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }

    // 功率调节
    adjustPower() {
        const powerValue = document.getElementById('powerAdjust').value;
        if (!powerValue) {
            this.showToast('请输入功率值', 'warning');
            return;
        }

        // 验证功率值范围
        if (powerValue < 0 || powerValue > 200) {
            this.showToast('功率值必须在0-200kW范围内', 'warning');
            return;
        }

        this.addLog(`开始调整功率至 ${powerValue}kW`);
        
        // 更新UI显示
        const progressBar = document.querySelector('.progress-bar');
        const utilizationText = document.querySelector('small.text-muted');
        const powerDisplay = document.querySelector('h4.mb-0');
        
        // 模拟渐进式调整
        let currentPower = parseInt(powerDisplay.textContent);
        const targetPower = parseInt(powerValue);
        const step = (targetPower - currentPower) / 10;
        
        let i = 0;
        const interval = setInterval(() => {
            if (i >= 10) {
                clearInterval(interval);
                this.addLog(`功率已成功调整至 ${powerValue}kW`);
                this.showToast('功率调节完成', 'success');
                
                // 更新系统参数
                document.getElementById('maxPowerLimit').value = powerValue;
                this.currentStrategy.powerLimits.max = parseFloat(powerValue);
                
                // 触发功率变化事件
                this.dispatchEvent('powerChanged', { 
                    value: powerValue,
                    timestamp: new Date().toISOString()
                });
                
                return;
            }
            
            currentPower += step;
            const utilization = (currentPower / 200 * 100).toFixed(1);
            
            powerDisplay.textContent = Math.round(currentPower) + ' MW';
            progressBar.style.width = utilization + '%';
            utilizationText.textContent = '利用率: ' + utilization + '%';
            
            i++;
        }, 200);
    }

    // 价格调整
    adjustPrice() {
        const priceValue = document.getElementById('priceAdjust').value;
        if (!priceValue) {
            this.showToast('请输入价格', 'warning');
            return;
        }

        // 验证价格范围
        if (priceValue < 0.1 || priceValue > 5) {
            this.showToast('价格必须在0.1-5元/kWh范围内', 'warning');
            return;
        }

        this.addLog(`开始调整价格至 ${priceValue}元/kWh`);

        // 更新价格显示
        const priceElements = document.querySelectorAll('.current-price');
        const oldPrice = parseFloat(priceElements[0].textContent.replace('¥', ''));
        const priceDiff = priceValue - oldPrice;

        // 模拟价格渐变
        let i = 0;
        const interval = setInterval(() => {
            if (i >= 10) {
                clearInterval(interval);
                this.addLog(`价格已成功调整至 ${priceValue}元/kWh`);
                this.showToast('价格调整完成', 'success');
                
                // 更新价格相关显示
                priceElements.forEach(el => {
                    el.textContent = '¥' + parseFloat(priceValue).toFixed(2);
                });
                
                // 触发价格更新事件
                this.dispatchEvent('priceUpdate', {
                    oldPrice: oldPrice,
                    newPrice: parseFloat(priceValue),
                    timestamp: new Date().toISOString()
                });
                
                // 添加到价格历史
                this.addPriceHistory(oldPrice, parseFloat(priceValue));
                return;
            }
            
            const currentPrice = oldPrice + (priceDiff * (i / 10));
            priceElements.forEach(el => {
                el.textContent = '¥' + currentPrice.toFixed(2);
            });
            
            i++;
        }, 200);
    }

    // 限流控制
    limitCurrent() {
        const currentValue = document.getElementById('currentLimit').value;
        if (!currentValue) {
            this.showToast('请输入电流值', 'warning');
            return;
        }

        // 验证电流范围
        if (currentValue < 10 || currentValue > 400) {
            this.showToast('电流值必须在10-400A范围内', 'warning');
            return;
        }

        this.addLog(`开始设置电流限制为 ${currentValue}A`);

        // 模拟电流限制过程
        const steps = [
            { message: '正在检查系统状态...', progress: 20 },
            { message: '验证安全参数...', progress: 40 },
            { message: '调整充电桩设置...', progress: 60 },
            { message: '同步限流配置...', progress: 80 },
            { message: '完成限流设置', progress: 100 }
        ];

        let currentStep = 0;
        const processStep = () => {
            if (currentStep >= steps.length) {
                this.addLog(`电流限制已设置为 ${currentValue}A`);
                this.showToast('限流控制设置完成', 'success');
                
                // 更新系统状态
                this.updateSystemStatus('currentLimit', currentValue);
                
                // 触发限流事件
                this.dispatchEvent('currentLimited', {
                    value: currentValue,
                    timestamp: new Date().toISOString()
                });
                
                return;
            }

            const step = steps[currentStep];
            this.addLog(step.message);
            
            // 更新进度条
            const progressBar = document.querySelector('.progress-bar');
            if (progressBar) {
                progressBar.style.width = step.progress + '%';
            }

            currentStep++;
            setTimeout(processStep, 800);
        };

        processStep();
    }

    // 切换运行模式
    switchMode(mode) {
        const modeNames = {
            normal: '正常模式',
            eco: '节能模式',
            fast: '快充模式',
            maintenance: '维护模式'
        };

        this.addLog(`切换至${modeNames[mode]}`);

        // 获取模式特定配置
        const modeConfig = this.getModeConfig(mode);
        
        // 模拟模式切换过程
        const steps = [
            { message: '保存当前配置...', progress: 20 },
            { message: '加载新模式参数...', progress: 40 },
            { message: '调整系统设置...', progress: 60 },
            { message: '更新充电策略...', progress: 80 },
            { message: '完成模式切换', progress: 100 }
        ];

        let currentStep = 0;
        const processStep = () => {
            if (currentStep >= steps.length) {
                this.addLog(`已成功切换至${modeNames[mode]}`);
                this.showToast(`已切换至${modeNames[mode]}`, 'success');
                
                // 应用模式特定设置
                this.applyModeSettings(modeConfig);
                
                // 触发模式切换事件
                this.dispatchEvent('modeChanged', {
                    mode: mode,
                    config: modeConfig,
                    timestamp: new Date().toISOString()
                });
                
                return;
            }

            const step = steps[currentStep];
            this.addLog(step.message);
            currentStep++;
            setTimeout(processStep, 600);
        };

        processStep();
    }

    // 获取模式配置
    getModeConfig(mode) {
        const configs = {
            normal: {
                maxPower: 120,
                response: 3,
                buffer: 5,
                currentLimit: 200,
                priceAdjust: 1.0
            },
            eco: {
                maxPower: 90,
                response: 5,
                buffer: 8,
                currentLimit: 150,
                priceAdjust: 0.8
            },
            fast: {
                maxPower: 150,
                response: 2,
                buffer: 3,
                currentLimit: 300,
                priceAdjust: 1.2
            },
            maintenance: {
                maxPower: 60,
                response: 8,
                buffer: 10,
                currentLimit: 100,
                priceAdjust: 0.5
            }
        };
        
        return configs[mode] || configs.normal;
    }

    // 应用模式设置
    applyModeSettings(config) {
        // 更新UI显示
        document.getElementById('maxPowerLimit').value = config.maxPower;
        document.getElementById('responseTime').value = config.response;
        document.getElementById('bufferTime').value = config.buffer;
        
        // 更新功率显示
        const powerDisplay = document.querySelector('h4.mb-0');
        if (powerDisplay) {
            powerDisplay.textContent = config.maxPower + ' MW';
        }
        
        // 更新进度条
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            const utilization = (config.maxPower / 200 * 100).toFixed(1);
            progressBar.style.width = utilization + '%';
        }
        
        // 更新系统状态
        this.currentStrategy = {
            ...this.currentStrategy,
            powerLimits: {
                max: config.maxPower,
                min: config.maxPower * 0.3
            },
            timing: {
                response: config.response,
                buffer: config.buffer
            }
        };
    }

    // 添加价格历史
    addPriceHistory(oldPrice, newPrice) {
        const priceHistory = this.priceHistory || [];
        priceHistory.push({
            timestamp: new Date().toISOString(),
            oldPrice: oldPrice,
            newPrice: newPrice,
            change: ((newPrice - oldPrice) / oldPrice * 100).toFixed(2) + '%'
        });
        
        // 保持最近100条记录
        if (priceHistory.length > 100) {
            priceHistory.shift();
        }
        
        this.priceHistory = priceHistory;
    }

    // 批量调度
    batchDispatch() {
        const modal = new bootstrap.Modal(document.getElementById('dispatchModal'));
        modal.show();

        // 填充设备列表
        this.populateDeviceList();

        // 绑定确认按钮事件
        document.getElementById('confirmDispatch').onclick = () => {
            const type = document.getElementById('dispatchType').value;
            const value = document.getElementById('dispatchValue').value;
            const devices = Array.from(document.getElementById('targetDevice').selectedOptions).map(opt => opt.value);

            if (!devices.length || !value) {
                this.showToast('请选择目标设备并输入调度值', 'warning');
                return;
            }

            this.addLog(`开始批量调度: ${type} - ${devices.length}个设备`);
            this.simulateOperation('批量调度', () => {
                this.addLog(`批量调度完成: ${devices.length}个设备已更新`);
                modal.hide();
            });
        };
    }

    // 负载优化
    optimizeLoad() {
        this.addLog('开始执行负载优化...');
        this.simulateOperation('负载优化', () => {
            // 模拟优化过程
            const optimizations = [
                { time: 1000, message: '分析当前负载分布...' },
                { time: 2000, message: '计算最优分配方案...' },
                { time: 3000, message: '调整充电桩功率分配...' },
                { time: 4000, message: '优化完成，负载均衡度提升15%' }
            ];

            optimizations.forEach((opt, index) => {
                setTimeout(() => {
                    this.addLog(opt.message);
                    if (index === optimizations.length - 1) {
                        this.showToast('负载优化完成', 'success');
                    }
                }, opt.time);
            });
        });
    }

    // 应急模式
    emergencyMode() {
        const confirmResult = confirm('确定要启动应急模式吗？这将影响所有充电设备的运行状态。');
        if (!confirmResult) return;

        this.addLog('正在启动应急模式...', 'warning');
        this.simulateOperation('应急模式', () => {
            // 执行应急预案
            const emergencySteps = [
                { time: 1000, action: () => this.addLog('降低所有充电桩功率至安全水平', 'warning') },
                { time: 2000, action: () => this.addLog('启动备用电源系统', 'warning') },
                { time: 3000, action: () => this.addLog('切换至应急调度策略', 'warning') },
                { time: 4000, action: () => {
                    this.addLog('应急模式已完全启动', 'success');
                    this.showToast('应急模式已启动', 'warning');
                    // 更新UI状态
                    document.getElementById('operationMode').value = 'maintenance';
                    this.currentStrategy.runningMode = 'maintenance';
                }}
            ];

            emergencySteps.forEach(step => {
                setTimeout(step.action, step.time);
            });
        });
    }

    // 模拟操作执行
    simulateOperation(operationType, successCallback) {
        this.showToast(`${operationType}执行中...`, 'info');
        
        setTimeout(() => {
            const success = Math.random() > 0.1; // 90%成功率
            if (success) {
                successCallback();
                this.showToast(`${operationType}成功`, 'success');
            } else {
                this.addLog(`${operationType}失败，请重试`, 'error');
                this.showToast(`${operationType}失败`, 'danger');
            }
        }, 1500);
    }

    // 填充设备列表
    populateDeviceList() {
        const deviceSelect = document.getElementById('targetDevice');
        deviceSelect.innerHTML = ''; // 清空现有选项

        // 添加示例设备
        const devices = [
            { id: 'CS001', name: '高新区A站快充桩#1' },
            { id: 'CS002', name: '高新区A站快充桩#2' },
            { id: 'CS003', name: '天府新区B站快充桩#1' },
            { id: 'CS004', name: '天府新区B站快充桩#2' },
            { id: 'CS005', name: '武侯区C站快充桩#1' }
        ];

        devices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.id;
            option.textContent = device.name;
            deviceSelect.appendChild(option);
        });
    }

    // 更新系统状态
    updateSystemStatus(key, value) {
        // 这里可以添加更多状态更新逻辑
        this.currentStrategy[key] = value;
        this.addLog(`系统${key}已更新为${value}`);
    }

    // 触发自定义事件
    dispatchEvent(eventName, data) {
        const event = new CustomEvent(eventName, { detail: data });
        document.dispatchEvent(event);
    }
}

// 全局函数定义
window.saveStrategy = function() {
    window.dispatchStrategy.saveStrategy();
};

window.applyStrategy = function() {
    window.dispatchStrategy.applyStrategy();
};

window.clearLog = function() {
    window.dispatchStrategy.clearLogs();
};

window.adjustPower = function() {
    window.dispatchStrategy.adjustPower();
};

window.adjustPrice = function() {
    window.dispatchStrategy.adjustPrice();
};

window.limitCurrent = function() {
    window.dispatchStrategy.limitCurrent();
};

window.switchMode = function(mode) {
    window.dispatchStrategy.switchMode(mode);
};

window.batchDispatch = function() {
    window.dispatchStrategy.batchDispatch();
};

window.optimizeLoad = function() {
    window.dispatchStrategy.optimizeLoad();
};

window.emergencyMode = function() {
    window.dispatchStrategy.emergencyMode();
};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    window.dispatchStrategy = new DispatchStrategy();
}); 