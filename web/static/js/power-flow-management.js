/**
 * 电力流程管理类
 * 处理发电、储电、用电的全流程调度
 */
class PowerFlowManagement {
    constructor() {
        // 防止重复初始化
        if (window.powerFlowManagementInstance) {
            return window.powerFlowManagementInstance;
        }
        window.powerFlowManagementInstance = this;

        // 初始化状态
        this.state = {
            // 发电系统状态
            generation: {
                wind: {
                    capacity: 500,      // 总装机容量(MW)
                    current: 320.5,     // 当前发电量(MW)
                    available: 450,     // 当前可用容量(MW)
                    efficiency: 0.85,   // 当前效率
                    units: [
                        { id: 'WT001', power: 45.5, status: 'running' },
                        { id: 'WT002', power: 42.8, status: 'running' },
                        // ... 更多风机
                    ]
                },
                solar: {
                    capacity: 600,
                    current: 425.8,
                    available: 520,
                    efficiency: 0.92,
                    arrays: [
                        { id: 'PV001', power: 85.2, status: 'running' },
                        { id: 'PV002', power: 82.5, status: 'running' },
                        // ... 更多光伏阵列
                    ]
                }
            },
            // 储能系统状态
            storage: {
                capacity: 200,          // 总储能容量(MWh)
                current: 110.0,         // 当前储能量(MWh)
                charging: 25.5,         // 充电功率(MW)
                discharging: 0,         // 放电功率(MW)
                soc: 0.55,             // 当前充电状态
                units: [
                    { id: 'ES001', capacity: 50, soc: 0.6, status: 'charging' },
                    { id: 'ES002', capacity: 50, soc: 0.5, status: 'standby' },
                    // ... 更多储能单元
                ]
            },
            // 负荷系统状态
            load: {
                total: 856.3,          // 总负荷(MW)
                peak: 1000,            // 峰值负荷(MW)
                valley: 400,           // 谷值负荷(MW)
                forecast: [            // 未来24小时负荷预测
                    { time: '00:00', value: 450 },
                    { time: '01:00', value: 420 },
                    // ... 更多预测数据
                ],
                categories: {
                    industrial: 450.2,  // 工业负荷(MW)
                    commercial: 280.5,  // 商业负荷(MW)
                    residential: 125.6  // 居民负荷(MW)
                }
            },
            // 电网状态
            grid: {
                frequency: 50.02,      // 系统频率(Hz)
                voltage: {
                    a: 230.5,          // A相电压(V)
                    b: 231.2,          // B相电压(V)
                    c: 230.8           // C相电压(V)
                },
                powerFactor: 0.95,     // 功率因数
                losses: 12.5           // 线损率(%)
            }
        };

        // 绑定事件处理器
        this.bindEventHandlers();
        
        // 启动实时监控
        this.startRealTimeMonitoring();
    }

    /**
     * 绑定事件处理器
     */
    bindEventHandlers() {
        // 监听发电调度事件
        document.querySelectorAll('[data-generation-control]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const type = e.target.dataset.generationType;
                const action = e.target.dataset.generationAction;
                this.handleGenerationControl(type, action);
            });
        });

        // 监听储能调度事件
        document.querySelectorAll('[data-storage-control]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.storageAction;
                this.handleStorageControl(action);
            });
        });

        // 监听负荷调度事件
        document.querySelectorAll('[data-load-control]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.loadAction;
                this.handleLoadControl(action);
            });
        });
    }

    /**
     * 处理发电控制
     */
    handleGenerationControl(type, action) {
        switch(action) {
            case 'increase':
                this.adjustGenerationOutput(type, 10); // 增加10MW
                break;
            case 'decrease':
                this.adjustGenerationOutput(type, -10); // 减少10MW
                break;
            case 'optimize':
                this.optimizeGeneration(type);
                break;
            case 'emergency':
                this.emergencyShutdown(type);
                break;
        }
    }

    /**
     * 处理储能控制
     */
    handleStorageControl(action) {
        switch(action) {
            case 'charge':
                this.startCharging();
                break;
            case 'discharge':
                this.startDischarging();
                break;
            case 'standby':
                this.setStorageStandby();
                break;
            case 'optimize':
                this.optimizeStorage();
                break;
        }
    }

    /**
     * 处理负荷控制
     */
    handleLoadControl(action) {
        switch(action) {
            case 'shed':
                this.performLoadShedding();
                break;
            case 'limit':
                this.setLoadLimit();
                break;
            case 'restore':
                this.restoreLoad();
                break;
            case 'optimize':
                this.optimizeLoad();
                break;
        }
    }

    /**
     * 调整发电输出
     */
    adjustGenerationOutput(type, delta) {
        const system = this.state.generation[type];
        if (!system) return;

        const newOutput = system.current + delta;
        if (newOutput < 0 || newOutput > system.capacity) {
            this.showToast('warning', '目标功率超出系统容量范围');
            return;
        }

        system.current = newOutput;
        this.updateSystemDisplay();
        this.addOperationLog('generation', `${type}发电量调整至${newOutput.toFixed(1)}MW`);
    }

    /**
     * 优化发电配置
     */
    optimizeGeneration(type) {
        const system = this.state.generation[type];
        if (!system) return;

        // 根据当前负荷和天气条件优化发电配置
        const optimizedPower = this.calculateOptimalPower(type);
        system.current = optimizedPower;

        this.updateSystemDisplay();
        this.addOperationLog('generation', `${type}发电优化完成，当前功率${optimizedPower.toFixed(1)}MW`);
    }

    /**
     * 启动储能充电
     */
    startCharging() {
        if (this.state.storage.soc >= 0.95) {
            this.showToast('warning', '储能系统接近满容量');
            return;
        }

        const chargingPower = this.calculateOptimalChargingPower();
        this.state.storage.charging = chargingPower;
        this.state.storage.discharging = 0;

        this.updateSystemDisplay();
        this.addOperationLog('storage', `开始充电，充电功率${chargingPower.toFixed(1)}MW`);
    }

    /**
     * 启动储能放电
     */
    startDischarging() {
        if (this.state.storage.soc <= 0.15) {
            this.showToast('warning', '储能系统电量不足');
            return;
        }

        const dischargingPower = this.calculateOptimalDischargingPower();
        this.state.storage.charging = 0;
        this.state.storage.discharging = dischargingPower;

        this.updateSystemDisplay();
        this.addOperationLog('storage', `开始放电，放电功率${dischargingPower.toFixed(1)}MW`);
    }

    /**
     * 执行负荷切除
     */
    performLoadShedding() {
        const shedAmount = this.calculateLoadShedding();
        this.state.load.total -= shedAmount;

        this.updateSystemDisplay();
        this.addOperationLog('load', `执行负荷切除${shedAmount.toFixed(1)}MW`);
    }

    /**
     * 计算最优充电功率
     */
    calculateOptimalChargingPower() {
        // 考虑当前发电余量和储能系统状态
        const availablePower = this.getAvailableChargingPower();
        const maxChargingPower = this.state.storage.capacity * 0.2; // 最大充电功率为容量的20%
        return Math.min(availablePower, maxChargingPower);
    }

    /**
     * 计算最优放电功率
     */
    calculateOptimalDischargingPower() {
        // 考虑当前负荷需求和储能系统状态
        const requiredPower = this.getRequiredDischargingPower();
        const maxDischargingPower = this.state.storage.capacity * 0.3; // 最大放电功率为容量的30%
        return Math.min(requiredPower, maxDischargingPower);
    }

    /**
     * 计算负荷切除量
     */
    calculateLoadShedding() {
        // 根据系统状态计算需要切除的负荷量
        const overload = this.state.load.total - this.getTotalAvailablePower();
        return Math.max(overload, 0);
    }

    /**
     * 获取系统总可用功率
     */
    getTotalAvailablePower() {
        return this.state.generation.wind.available + 
               this.state.generation.solar.available + 
               this.state.storage.current;
    }

    /**
     * 更新系统显示
     */
    updateSystemDisplay() {
        // 更新功率显示
        this.updatePowerDisplay();
        // 更新储能状态
        this.updateStorageDisplay();
        // 更新负荷显示
        this.updateLoadDisplay();
        // 更新系统状态图表
        this.updateSystemCharts();
    }

    /**
     * 添加操作日志
     */
    addOperationLog(type, message) {
        const logEntry = {
            timestamp: new Date(),
            type: type,
            message: message
        };

        // 添加到历史记录
        if (window.powerDispatchManagement) {
            window.powerDispatchManagement.addDispatchHistory(logEntry);
        }
    }

    /**
     * 启动实时监控
     */
    startRealTimeMonitoring() {
        setInterval(() => {
            this.updateRealTimeData();
        }, 5000); // 每5秒更新一次
    }

    /**
     * 更新实时数据
     */
    updateRealTimeData() {
        // 更新发电数据
        this.updateGenerationData();
        // 更新储能数据
        this.updateStorageData();
        // 更新负荷数据
        this.updateLoadData();
        // 更新系统显示
        this.updateSystemDisplay();
    }

    // ... 其他辅助方法
}

// 初始化电力流程管理
document.addEventListener('DOMContentLoaded', () => {
    window.powerFlowManagement = new PowerFlowManagement();
}); 