/**
 * 电力调度管理类
 */
class PowerDispatchManagement {
    constructor() {
        // 防止重复初始化
        if (window.powerDispatchManagementInstance) {
            return window.powerDispatchManagementInstance;
        }
        window.powerDispatchManagementInstance = this;

        // 初始化状态
        this.state = {
            currentPower: {
                wind: 320.5,
                solar: 425.8,
                storage: 110.0,
                load: 856.3
            },
            deviceStatus: {
                windTurbines: { online: 20, offline: 2 },
                solarPanels: { online: 15, offline: 0 },
                storageUnits: { online: 4, maintenance: 1 }
            },
            alerts: []
        };

        // 绑定事件处理器
        this.bindEventHandlers();
    }

    /**
     * 绑定事件处理器
     */
    bindEventHandlers() {
        // 快速调度按钮
        document.querySelectorAll('.quick-dispatch .btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const type = e.currentTarget.textContent.includes('风电') ? 'wind' :
                            e.currentTarget.textContent.includes('光伏') ? 'solar' :
                            e.currentTarget.textContent.includes('储能') ? 'storage' : 'load';
                this.showQuickDispatchModal(type);
            });
        });

        // 应急控制按钮
        document.querySelector('.btn-danger').addEventListener('click', () => {
            this.showEmergencyControlModal();
        });

        // 新建计划按钮
        document.querySelector('.btn-primary[title="新建计划"]')?.addEventListener('click', () => {
            this.showNewPlanModal();
        });
    }

    /**
     * 显示快速调度模态框
     */
    showQuickDispatchModal(type) {
        const modalHtml = `
            <div class="modal fade" id="quickDispatchModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${this.getDispatchTitle(type)}调度</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <form id="dispatchForm">
                                <div class="mb-3">
                                    <label class="form-label">当前功率</label>
                                    <div class="input-group">
                                        <input type="text" class="form-control" value="${this.state.currentPower[type]}" readonly>
                                        <span class="input-group-text">MW</span>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">目标功率</label>
                                    <div class="input-group">
                                        <input type="number" class="form-control" id="targetPower" required>
                                        <span class="input-group-text">MW</span>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">调节速率</label>
                                    <select class="form-select" id="adjustRate">
                                        <option value="slow">缓慢 (1MW/min)</option>
                                        <option value="normal" selected>正常 (5MW/min)</option>
                                        <option value="fast">快速 (10MW/min)</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">备注说明</label>
                                    <textarea class="form-control" id="dispatchNote" rows="3"></textarea>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-light" data-bs-dismiss="modal">取消</button>
                            <button type="button" class="btn btn-primary" onclick="powerDispatchManagement.executeDispatch('${type}')">
                                执行调度
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // 添加模态框到页面
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('quickDispatchModal'));
        modal.show();

        // 模态框关闭时移除
        document.getElementById('quickDispatchModal').addEventListener('hidden.bs.modal', function () {
            this.remove();
        });
    }

    /**
     * 显示应急控制模态框
     */
    showEmergencyControlModal() {
        const modalHtml = `
            <div class="modal fade" id="emergencyControlModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header bg-danger text-white">
                            <h5 class="modal-title">应急控制</h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="alert alert-warning">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                请谨慎操作！应急控制将立即执行，可能影响系统正常运行。
                            </div>
                            <form id="emergencyForm">
                                <div class="mb-3">
                                    <label class="form-label">应急措施</label>
                                    <select class="form-select" id="emergencyAction" required>
                                        <option value="">选择应急措施...</option>
                                        <option value="shutdown">紧急停机</option>
                                        <option value="loadShedding">负荷切除</option>
                                        <option value="islandOperation">孤岛运行</option>
                                        <option value="blackStart">黑启动</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">影响范围</label>
                                    <select class="form-select" id="emergencyScope" required>
                                        <option value="all">全站</option>
                                        <option value="wind">风电场</option>
                                        <option value="solar">光伏场</option>
                                        <option value="storage">储能站</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">原因说明</label>
                                    <textarea class="form-control" id="emergencyReason" rows="3" required></textarea>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-light" data-bs-dismiss="modal">取消</button>
                            <button type="button" class="btn btn-danger" onclick="powerDispatchManagement.executeEmergencyControl()">
                                确认执行
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // 添加模态框到页面
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('emergencyControlModal'));
        modal.show();

        // 模态框关闭时移除
        document.getElementById('emergencyControlModal').addEventListener('hidden.bs.modal', function () {
            this.remove();
        });
    }

    /**
     * 显示新建计划模态框
     */
    showNewPlanModal() {
        const modalHtml = `
            <div class="modal fade" id="newPlanModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">新建调度计划</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <form id="planForm">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label class="form-label">计划类型</label>
                                            <select class="form-select" id="planType" required>
                                                <option value="">选择类型...</option>
                                                <option value="powerAdjustment">功率调整</option>
                                                <option value="storageDispatch">储能调度</option>
                                                <option value="loadManagement">负荷管理</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label class="form-label">目标设备</label>
                                            <select class="form-select" id="targetDevice" required>
                                                <option value="">选择设备...</option>
                                                <option value="windGroup1">风机组#1-5</option>
                                                <option value="solarArray1">光伏阵列A</option>
                                                <option value="storageUnit1">储能站#1</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label class="form-label">执行时间</label>
                                            <input type="datetime-local" class="form-control" id="executeTime" required>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label class="form-label">目标值</label>
                                            <div class="input-group">
                                                <input type="number" class="form-control" id="targetValue" required>
                                                <span class="input-group-text">MW</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">计划说明</label>
                                    <textarea class="form-control" id="planDescription" rows="3" required></textarea>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-light" data-bs-dismiss="modal">取消</button>
                            <button type="button" class="btn btn-primary" onclick="powerDispatchManagement.createDispatchPlan()">
                                创建计划
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // 添加模态框到页面
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('newPlanModal'));
        modal.show();

        // 模态框关闭时移除
        document.getElementById('newPlanModal').addEventListener('hidden.bs.modal', function () {
            this.remove();
        });
    }

    /**
     * 执行调度
     */
    executeDispatch(type) {
        const form = document.getElementById('dispatchForm');
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        const targetPower = parseFloat(document.getElementById('targetPower').value);
        const adjustRate = document.getElementById('adjustRate').value;
        const note = document.getElementById('dispatchNote').value;

        // 执行调度逻辑
        this.updatePowerValue(type, targetPower);
        
        // 添加到历史记录
        this.addDispatchHistory({
            type: type,
            target: targetPower,
            rate: adjustRate,
            note: note,
            time: new Date()
        });

        // 关闭模态框
        bootstrap.Modal.getInstance(document.getElementById('quickDispatchModal')).hide();

        // 显示成功提示
        this.showToast('success', '调度指令已下发');
    }

    /**
     * 执行应急控制
     */
    executeEmergencyControl() {
        const form = document.getElementById('emergencyForm');
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        const action = document.getElementById('emergencyAction').value;
        const scope = document.getElementById('emergencyScope').value;
        const reason = document.getElementById('emergencyReason').value;

        // 执行应急控制逻辑
        this.performEmergencyAction(action, scope);

        // 添加到历史记录
        this.addDispatchHistory({
            type: 'emergency',
            action: action,
            scope: scope,
            reason: reason,
            time: new Date()
        });

        // 关闭模态框
        bootstrap.Modal.getInstance(document.getElementById('emergencyControlModal')).hide();

        // 显示警告提示
        this.showToast('warning', '应急控制已执行');
    }

    /**
     * 创建调度计划
     */
    createDispatchPlan() {
        const form = document.getElementById('planForm');
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        const planData = {
            id: 'PLAN' + Date.now(),
            type: document.getElementById('planType').value,
            target: document.getElementById('targetDevice').value,
            executeTime: document.getElementById('executeTime').value,
            targetValue: document.getElementById('targetValue').value,
            description: document.getElementById('planDescription').value,
            status: 'pending'
        };

        // 添加到计划列表
        this.addDispatchPlan(planData);

        // 关闭模态框
        bootstrap.Modal.getInstance(document.getElementById('newPlanModal')).hide();

        // 显示成功提示
        this.showToast('success', '调度计划已创建');
    }

    /**
     * 更新功率值
     */
    updatePowerValue(type, value) {
        this.state.currentPower[type] = value;
        // 更新显示
        this.updatePowerDisplay();
    }

    /**
     * 执行应急操作
     */
    performEmergencyAction(action, scope) {
        switch(action) {
            case 'shutdown':
                this.state.currentPower[scope] = 0;
                break;
            case 'loadShedding':
                this.state.currentPower.load *= 0.7; // 降低30%负荷
                break;
            case 'islandOperation':
                // 实现孤岛运行逻辑
                break;
            case 'blackStart':
                // 实现黑启动逻辑
                break;
        }
        // 更新显示
        this.updatePowerDisplay();
    }

    /**
     * 更新功率显示
     */
    updatePowerDisplay() {
        // 更新功率卡片显示
        Object.entries(this.state.currentPower).forEach(([type, value]) => {
            const element = document.querySelector(`[data-power-type="${type}"]`);
            if (element) {
                element.textContent = value.toFixed(1);
            }
        });

        // 更新图表
        if (window.powerDispatchCharts) {
            window.powerDispatchCharts.updatePowerCurve(this.state.currentPower);
        }
    }

    /**
     * 添加调度历史记录
     */
    addDispatchHistory(record) {
        const historyContainer = document.querySelector('.activity-timeline');
        if (!historyContainer) return;

        const timeString = new Date(record.time).toLocaleTimeString();
        const historyHtml = `
            <div class="d-flex align-items-start">
                <div class="flex-shrink-0">
                    <div class="avatar-xs">
                        <div class="avatar-title rounded-circle bg-soft-${this.getHistoryColor(record.type)} text-${this.getHistoryColor(record.type)}">
                            <i class="fas ${this.getHistoryIcon(record.type)}"></i>
                        </div>
                    </div>
                </div>
                <div class="flex-grow-1 ms-3">
                    <h6 class="mb-1">${this.getHistoryTitle(record)}</h6>
                    <p class="text-muted mb-0">${timeString} - ${this.getHistoryDescription(record)}</p>
                </div>
                <div class="flex-shrink-0">
                    <small class="text-muted">刚刚</small>
                </div>
            </div>
        `;

        historyContainer.insertAdjacentHTML('afterbegin', historyHtml);
    }

    /**
     * 添加调度计划
     */
    addDispatchPlan(plan) {
        const tbody = document.querySelector('.table tbody');
        if (!tbody) return;

        const planHtml = `
            <tr>
                <td>${plan.id}</td>
                <td>${this.getPlanTypeText(plan.type)}</td>
                <td>${this.getTargetDeviceText(plan.target)}</td>
                <td>${plan.executeTime}</td>
                <td><span class="badge bg-info">待执行</span></td>
                <td>
                    <div class="d-flex gap-2">
                        <button class="btn btn-sm btn-soft-primary" onclick="powerDispatchManagement.viewPlanDetails('${plan.id}')">详情</button>
                        <button class="btn btn-sm btn-soft-danger" onclick="powerDispatchManagement.cancelPlan('${plan.id}')">取消</button>
                    </div>
                </td>
            </tr>
        `;

        tbody.insertAdjacentHTML('afterbegin', planHtml);
    }

    /**
     * 显示提示消息
     */
    showToast(type, message) {
        const toastHtml = `
            <div class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        const toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            document.body.insertAdjacentHTML('beforeend', '<div class="toast-container position-fixed top-0 end-0 p-3"></div>');
        }

        document.querySelector('.toast-container').insertAdjacentHTML('beforeend', toastHtml);
        const toast = new bootstrap.Toast(document.querySelector('.toast-container .toast:last-child'));
        toast.show();
    }

    // 辅助方法
    getDispatchTitle(type) {
        const titles = {
            wind: '风电',
            solar: '光伏',
            storage: '储能',
            load: '负荷'
        };
        return titles[type] || '';
    }

    getHistoryColor(type) {
        const colors = {
            wind: 'primary',
            solar: 'success',
            storage: 'warning',
            load: 'info',
            emergency: 'danger'
        };
        return colors[type] || 'secondary';
    }

    getHistoryIcon(type) {
        const icons = {
            wind: 'fa-wind',
            solar: 'fa-sun',
            storage: 'fa-battery-half',
            load: 'fa-plug',
            emergency: 'fa-exclamation-triangle'
        };
        return icons[type] || 'fa-cog';
    }

    getHistoryTitle(record) {
        if (record.type === 'emergency') {
            return `应急控制 - ${this.getEmergencyActionText(record.action)}`;
        }
        return `${this.getDispatchTitle(record.type)}功率调整`;
    }

    getHistoryDescription(record) {
        if (record.type === 'emergency') {
            return `范围: ${this.getEmergencyScopeText(record.scope)}`;
        }
        return `目标功率: ${record.target}MW`;
    }

    getPlanTypeText(type) {
        const types = {
            powerAdjustment: '功率调整',
            storageDispatch: '储能调度',
            loadManagement: '负荷管理'
        };
        return types[type] || type;
    }

    getTargetDeviceText(target) {
        const devices = {
            windGroup1: '风机组#1-5',
            solarArray1: '光伏阵列A',
            storageUnit1: '储能站#1'
        };
        return devices[target] || target;
    }

    getEmergencyActionText(action) {
        const actions = {
            shutdown: '紧急停机',
            loadShedding: '负荷切除',
            islandOperation: '孤岛运行',
            blackStart: '黑启动'
        };
        return actions[action] || action;
    }

    getEmergencyScopeText(scope) {
        const scopes = {
            all: '全站',
            wind: '风电场',
            solar: '光伏场',
            storage: '储能站'
        };
        return scopes[scope] || scope;
    }
}

// 初始化电力调度管理
document.addEventListener('DOMContentLoaded', () => {
    window.powerDispatchManagement = new PowerDispatchManagement();
}); 