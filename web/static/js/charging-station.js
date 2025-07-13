// 充电桩管理系统功能实现

class ChargingStationController {
    constructor() {
        // 单例模式
        if (window.chargingStationController) {
            return window.chargingStationController;
        }
        window.chargingStationController = this;

        // 初始化属性
        this.stations = new Map(); // 存储充电站信息
        this.chargingQueues = new Map(); // 充电队列
        this.scheduleTimers = new Map(); // 调度定时器
        this.faultAlerts = new Set(); // 故障告警
        this.reservations = new Map(); // 预约信息

        // 初始化
        this.initialize();
    }

    async initialize() {
        try {
            await this.loadStationData();
            this.initializeEventListeners();
            this.startMonitoring();
            this.renderStationList();
        } catch (error) {
            console.error('初始化充电站控制器失败:', error);
        }
    }

    // 渲染充电站列表
    renderStationList() {
        const listContainer = document.querySelector('.charging-station-list');
        if (!listContainer) return;

        const stationData = [
            { id: "CS001", name: "成都高新区A站", location: "成都市高新区", status: "正常", type: "快充站", power: "120kW", ports: "10/12" },
            { id: "CS002", name: "成都天府新区B站", location: "成都市天府新区", status: "正常", type: "快充站", power: "90kW", ports: "8/10" },
            { id: "CS003", name: "成都武侯区C站", location: "成都市武侯区", status: "维护中", type: "快充站", power: "150kW", ports: "12/15" },
            { id: "CS004", name: "绵阳涪城区A站", location: "绵阳市涪城区", status: "正常", type: "快充站", power: "100kW", ports: "6/8" },
            { id: "CS005", name: "德阳旌阳区B站", location: "德阳市旌阳区", status: "正常", type: "快充站", power: "80kW", ports: "5/6" }
        ];

        const html = stationData.map(station => `
            <div class="station-card">
                <div class="station-header">
                    <h6>${station.name}</h6>
                    <span class="badge bg-${station.status === '正常' ? 'success' : 'warning'}">${station.status}</span>
                </div>
                <div class="station-body">
                    <div class="info-row">
                        <span class="label">位置：</span>
                        <span class="value">${station.location}</span>
                    </div>
                    <div class="info-row">
                        <span class="label">类型：</span>
                        <span class="value">${station.type}</span>
                    </div>
                    <div class="info-row">
                        <span class="label">功率：</span>
                        <span class="value">${station.power}</span>
                    </div>
                    <div class="info-row">
                        <span class="label">可用端口：</span>
                        <span class="value">${station.ports}</span>
                    </div>
                </div>
                <div class="station-footer">
                    <button class="btn btn-sm btn-primary" onclick="showStationDetails('${station.id}')">
                        <i class="ri-eye-line"></i> 查看详情
                    </button>
                    <button class="btn btn-sm btn-info" onclick="showReservation('${station.id}')">
                        <i class="ri-calendar-line"></i> 预约
                    </button>
                    <button class="btn btn-sm btn-warning" onclick="showMaintenance('${station.id}')">
                        <i class="ri-tools-line"></i> 维护
                    </button>
                </div>
            </div>
        `).join('');

        listContainer.innerHTML = html;
    }

    // 显示充电站详情
    showStationDetails(stationId) {
        const station = this.stations.get(stationId);
        if (!station) return;

        const modal = new bootstrap.Modal(document.getElementById('monitoringModal'));
        document.getElementById('monitoringStationName').textContent = station.name;
        
        // 更新实时数据
        const monitoringData = document.querySelector('.monitoring-data');
        monitoringData.innerHTML = `
            <div class="d-flex justify-content-between mb-3">
                <span>状态</span>
                <span class="badge bg-${station.status === '正常' ? 'success' : 'warning'}">${station.status}</span>
            </div>
            <div class="d-flex justify-content-between mb-3">
                <span>充电功率</span>
                <span>${station.power}</span>
            </div>
            <div class="d-flex justify-content-between">
                <span>可用端口</span>
                <span>${station.ports}</span>
            </div>
        `;
        
        modal.show();
    }

    // 1. 充电桩调度控制功能
    
    /**
     * 调整充电功率
     * @param {string} stationId 充电站ID
     * @param {string} portId 充电端口ID
     * @param {number} power 目标功率(kW)
     * @param {string} reason 调整原因
     */
    async adjustPower(stationId, portId, power, reason) {
        try {
            const station = this.stations.get(stationId);
            if (!station) throw new Error('充电站不存在');

            const port = station.ports.get(portId);
            if (!port) throw new Error('充电端口不存在');

            // 验证功率范围
            if (power < port.minPower || power > port.maxPower) {
                throw new Error(`功率必须在 ${port.minPower}kW - ${port.maxPower}kW 之间`);
            }

            // 记录调整前的功率
            const previousPower = port.currentPower;

            // 执行功率调整
            await this._sendControlCommand(stationId, portId, 'adjustPower', { power });
            
            // 更新端口状态
            port.currentPower = power;
            port.lastAdjustment = {
                time: new Date(),
                from: previousPower,
                to: power,
                reason
            };

            // 触发事件
            this._emitEvent('powerAdjusted', {
                stationId,
                portId,
                previousPower,
                currentPower: power,
                reason
            });

            return true;
        } catch (error) {
            console.error('调整充电功率失败:', error);
            throw error;
        }
    }

    /**
     * 启动或停止充电
     * @param {string} stationId 充电站ID
     * @param {string} portId 充电端口ID
     * @param {boolean} start true表示启动,false表示停止
     */
    async toggleCharging(stationId, portId, start) {
        try {
            const station = this.stations.get(stationId);
            if (!station) throw new Error('充电站不存在');

            const port = station.ports.get(portId);
            if (!port) throw new Error('充电端口不存在');

            // 检查端口状态
            if (start && port.status !== 'ready') {
                throw new Error('端口当前状态不允许启动充电');
            }
            if (!start && port.status !== 'charging') {
                throw new Error('端口当前状态不允许停止充电');
            }

            // 执行启停控制
            await this._sendControlCommand(stationId, portId, start ? 'startCharging' : 'stopCharging');

            // 更新端口状态
            port.status = start ? 'charging' : 'ready';
            port.lastStatusChange = new Date();

            // 触发事件
            this._emitEvent('chargingStateChanged', {
                stationId,
                portId,
                status: port.status
            });

            return true;
        } catch (error) {
            console.error('控制充电状态失败:', error);
            throw error;
        }
    }

    /**
     * 切换充电模式
     * @param {string} stationId 充电站ID
     * @param {string} portId 充电端口ID
     * @param {string} mode 充电模式(fast/normal/eco)
     */
    async switchChargingMode(stationId, portId, mode) {
        try {
            const station = this.stations.get(stationId);
            if (!station) throw new Error('充电站不存在');

            const port = station.ports.get(portId);
            if (!port) throw new Error('充电端口不存在');

            // 验证模式
            const validModes = ['fast', 'normal', 'eco'];
            if (!validModes.includes(mode)) {
                throw new Error('无效的充电模式');
            }

            // 执行模式切换
            await this._sendControlCommand(stationId, portId, 'switchMode', { mode });

            // 更新端口配置
            port.chargingMode = mode;
            port.lastModeSwitch = new Date();

            // 根据模式调整功率限制
            switch (mode) {
                case 'fast':
                    port.maxPower = port.specifications.maxPower;
                    break;
                case 'normal':
                    port.maxPower = port.specifications.maxPower * 0.8;
                    break;
                case 'eco':
                    port.maxPower = port.specifications.maxPower * 0.6;
                    break;
            }

            // 触发事件
            this._emitEvent('chargingModeChanged', {
                stationId,
                portId,
                mode,
                maxPower: port.maxPower
            });

            return true;
        } catch (error) {
            console.error('切换充电模式失败:', error);
            throw error;
        }
    }

    // 2. 负载均衡管理功能

    /**
     * 执行负载均衡
     * @param {string} stationId 充电站ID
     */
    async balanceLoad(stationId) {
        try {
            const station = this.stations.get(stationId);
            if (!station) throw new Error('充电站不存在');

            // 获取所有正在充电的端口
            const chargingPorts = Array.from(station.ports.values())
                .filter(port => port.status === 'charging');

            // 计算当前总负载
            const totalLoad = chargingPorts.reduce((sum, port) => sum + port.currentPower, 0);
            const averageLoad = totalLoad / chargingPorts.length;

            // 调整每个端口的功率以达到均衡
            for (const port of chargingPorts) {
                const deviation = port.currentPower - averageLoad;
                if (Math.abs(deviation) > 5) { // 如果偏差超过5kW则调整
                    const targetPower = averageLoad;
                    await this.adjustPower(
                        stationId,
                        port.id,
                        targetPower,
                        '负载均衡调整'
                    );
                }
            }

            // 触发事件
            this._emitEvent('loadBalanced', {
                stationId,
                totalLoad,
                averageLoad,
                portsAdjusted: chargingPorts.length
            });

            return true;
        } catch (error) {
            console.error('执行负载均衡失败:', error);
            throw error;
        }
    }

    /**
     * 峰谷时段功率调节
     * @param {string} stationId 充电站ID
     * @param {string} period 时段类型(peak/valley)
     */
    async adjustPeakValleyPower(stationId, period) {
        try {
            const station = this.stations.get(stationId);
            if (!station) throw new Error('充电站不存在');

            // 获取时段配置
            const periodConfig = {
                peak: {
                    maxTotalPower: station.specifications.maxTotalPower * 0.8,
                    portPowerLimit: 0.7
                },
                valley: {
                    maxTotalPower: station.specifications.maxTotalPower,
                    portPowerLimit: 1.0
                }
            }[period];

            if (!periodConfig) throw new Error('无效的时段类型');

            // 调整所有充电端口的功率限制
            for (const [portId, port] of station.ports) {
                const newMaxPower = port.specifications.maxPower * periodConfig.portPowerLimit;
                
                // 如果当前功率超过新的限制,则需要调整
                if (port.currentPower > newMaxPower) {
                    await this.adjustPower(
                        stationId,
                        portId,
                        newMaxPower,
                        `${period === 'peak' ? '峰' : '谷'}时段功率调整`
                    );
                }

                // 更新端口功率限制
                port.maxPower = newMaxPower;
            }

            // 更新站点配置
            station.currentPeriod = period;
            station.maxTotalPower = periodConfig.maxTotalPower;

            // 触发事件
            this._emitEvent('periodPowerAdjusted', {
                stationId,
                period,
                maxTotalPower: station.maxTotalPower
            });

            return true;
        } catch (error) {
            console.error('峰谷时段功率调节失败:', error);
            throw error;
        }
    }

    // 3. 故障处理功能

    /**
     * 处理故障告警
     * @param {string} stationId 充电站ID
     * @param {string} portId 充电端口ID
     * @param {Object} fault 故障信息
     */
    async handleFault(stationId, portId, fault) {
        try {
            const station = this.stations.get(stationId);
            if (!station) throw new Error('充电站不存在');

            const port = station.ports.get(portId);
            if (!port) throw new Error('充电端口不存在');

            // 记录故障信息
            const faultRecord = {
                id: `F${Date.now()}`,
                stationId,
                portId,
                type: fault.type,
                severity: fault.severity,
                description: fault.description,
                time: new Date(),
                status: 'pending',
                handlingSteps: []
            };

            // 根据故障严重程度采取措施
            switch (fault.severity) {
                case 'critical':
                    // 紧急停止充电
                    await this.toggleCharging(stationId, portId, false);
                    // 关闭端口
                    await this._sendControlCommand(stationId, portId, 'shutdown');
                    port.status = 'fault';
                    break;
                
                case 'warning':
                    // 降低充电功率
                    await this.adjustPower(
                        stationId,
                        portId,
                        port.currentPower * 0.5,
                        '故障预防性降功率'
                    );
                    break;
                
                case 'notice':
                    // 记录警告但不中断充电
                    faultRecord.handlingSteps.push({
                        time: new Date(),
                        action: '记录警告',
                        result: '继续监控'
                    });
                    break;
            }

            // 添加到故障记录
            this.faultAlerts.add(faultRecord);

            // 如果需要维修,创建工单
            if (fault.severity === 'critical' || fault.severity === 'warning') {
                await this.createMaintenanceOrder(stationId, portId, fault);
            }

            // 触发事件
            this._emitEvent('faultHandled', {
                stationId,
                portId,
                fault: faultRecord
            });

            return faultRecord;
        } catch (error) {
            console.error('处理故障告警失败:', error);
            throw error;
        }
    }

    /**
     * 创建维修工单
     * @param {string} stationId 充电站ID
     * @param {string} portId 充电端口ID
     * @param {Object} fault 故障信息
     */
    async createMaintenanceOrder(stationId, portId, fault) {
        try {
            const station = this.stations.get(stationId);
            if (!station) throw new Error('充电站不存在');

            const port = station.ports.get(portId);
            if (!port) throw new Error('充电端口不存在');

            // 创建工单
            const order = {
                id: `M${Date.now()}`,
                stationId,
                portId,
                faultId: fault.id,
                type: 'fault',
                priority: fault.severity === 'critical' ? 'high' : 'normal',
                status: 'created',
                createTime: new Date(),
                description: `故障类型: ${fault.type}\n故障描述: ${fault.description}`,
                assignee: null,
                steps: []
            };

            // 分配维修人员
            const assignee = await this._assignMaintainer(order);
            order.assignee = assignee;
            order.status = 'assigned';

            // 添加到工单列表
            station.maintenanceOrders.set(order.id, order);

            // 触发事件
            this._emitEvent('maintenanceOrderCreated', {
                stationId,
                portId,
                order
            });

            return order;
        } catch (error) {
            console.error('创建维修工单失败:', error);
            throw error;
        }
    }

    // 4. 预约管理功能

    /**
     * 创建充电预约
     * @param {string} stationId 充电站ID
     * @param {string} portId 充电端口ID
     * @param {Object} reservation 预约信息
     */
    async createReservation(stationId, portId, reservation) {
        try {
            const station = this.stations.get(stationId);
            if (!station) throw new Error('充电站不存在');

            const port = station.ports.get(portId);
            if (!port) throw new Error('充电端口不存在');

            // 验证时间段是否可用
            const isAvailable = await this._checkTimeSlotAvailability(
                stationId,
                portId,
                reservation.startTime,
                reservation.endTime
            );

            if (!isAvailable) {
                throw new Error('该时段已被预约');
            }

            // 创建预约记录
            const record = {
                id: `R${Date.now()}`,
                stationId,
                portId,
                userId: reservation.userId,
                startTime: new Date(reservation.startTime),
                endTime: new Date(reservation.endTime),
                status: 'confirmed',
                createTime: new Date(),
                power: reservation.power || port.specifications.maxPower,
                priority: reservation.priority || 'normal'
            };

            // 添加到预约列表
            if (!this.reservations.has(stationId)) {
                this.reservations.set(stationId, new Map());
            }
            this.reservations.get(stationId).set(record.id, record);

            // 设置预约提醒
            this._scheduleReservationReminder(record);

            // 触发事件
            this._emitEvent('reservationCreated', {
                stationId,
                portId,
                reservation: record
            });

            return record;
        } catch (error) {
            console.error('创建充电预约失败:', error);
            throw error;
        }
    }

    /**
     * 调整预约队列
     * @param {string} stationId 充电站ID
     */
    async optimizeReservationQueue(stationId) {
        try {
            const station = this.stations.get(stationId);
            if (!station) throw new Error('充电站不存在');

            const stationReservations = this.reservations.get(stationId);
            if (!stationReservations || stationReservations.size === 0) {
                return true;
            }

            // 获取所有待处理的预约
            const pendingReservations = Array.from(stationReservations.values())
                .filter(r => r.status === 'confirmed')
                .sort((a, b) => {
                    // 首先按优先级排序
                    if (a.priority !== b.priority) {
                        return a.priority === 'high' ? -1 : 1;
                    }
                    // 然后按开始时间排序
                    return a.startTime - b.startTime;
                });

            // 获取可用的充电端口
            const availablePorts = Array.from(station.ports.values())
                .filter(port => port.status === 'ready');

            // 重新分配端口
            for (const reservation of pendingReservations) {
                // 找到最适合的端口
                const bestPort = this._findBestPortForReservation(
                    availablePorts,
                    reservation
                );

                if (bestPort) {
                    // 更新预约信息
                    reservation.portId = bestPort.id;
                    // 标记端口为预约状态
                    bestPort.nextReservation = reservation;
                }
            }

            // 触发事件
            this._emitEvent('reservationQueueOptimized', {
                stationId,
                reservationsProcessed: pendingReservations.length
            });

            return true;
        } catch (error) {
            console.error('调整预约队列失败:', error);
            throw error;
        }
    }

    // 私有辅助方法

    /**
     * 发送控制命令
     * @private
     */
    async _sendControlCommand(stationId, portId, command, params = {}) {
        // 这里应该实现与充电桩通信的具体逻辑
        // 当前使用模拟实现
        return new Promise((resolve) => {
            setTimeout(() => {
                console.log('发送控制命令:', {
                    stationId,
                    portId,
                    command,
                    params
                });
                resolve(true);
            }, 500);
        });
    }

    /**
     * 分配维修人员
     * @private
     */
    async _assignMaintainer(order) {
        // 这里应该实现实际的人员分配逻辑
        // 当前返回模拟数据
        return {
            id: 'M001',
            name: '张工',
            phone: '13800138000'
        };
    }

    /**
     * 检查时间段可用性
     * @private
     */
    async _checkTimeSlotAvailability(stationId, portId, startTime, endTime) {
        const stationReservations = this.reservations.get(stationId);
        if (!stationReservations) return true;

        // 检查是否与现有预约冲突
        const hasConflict = Array.from(stationReservations.values())
            .some(r => {
                if (r.portId !== portId) return false;
                return (startTime < r.endTime && endTime > r.startTime);
            });

        return !hasConflict;
    }

    /**
     * 设置预约提醒
     * @private
     */
    _scheduleReservationReminder(reservation) {
        // 预约开始前15分钟提醒
        const reminderTime = new Date(reservation.startTime.getTime() - 15 * 60000);
        const now = new Date();

        if (reminderTime > now) {
            setTimeout(() => {
                this._emitEvent('reservationReminder', {
                    reservation,
                    timeToStart: '15分钟'
                });
            }, reminderTime - now);
        }
    }

    /**
     * 为预约找到最佳充电端口
     * @private
     */
    _findBestPortForReservation(availablePorts, reservation) {
        return availablePorts.find(port => 
            port.specifications.maxPower >= reservation.power &&
            !port.nextReservation
        );
    }

    /**
     * 触发事件
     * @private
     */
    _emitEvent(eventName, data) {
        const event = new CustomEvent(`charging-station-${eventName}`, {
            detail: data
        });
        window.dispatchEvent(event);
    }
}

// 初始化控制器
document.addEventListener('DOMContentLoaded', function() {
    window.chargingStationController = new ChargingStationController();
}); 