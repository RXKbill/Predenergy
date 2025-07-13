/**
 * 电力调度指令管理类
 */
class PowerDispatchCommands {
    constructor() {
        // 防止重复初始化
        if (window.powerDispatchCommandsInstance) {
            return window.powerDispatchCommandsInstance;
        }
        window.powerDispatchCommandsInstance = this;

        // 初始化状态
        this.state = {
            // 调度指令列表
            commands: new Map(),
            // 指令模板
            templates: {
                powerAdjustment: {
                    name: '功率调整指令',
                    template: '对{target}执行{action}调整，目标值{value}{unit}，执行时间{time}',
                    parameters: ['target', 'action', 'value', 'unit', 'time']
                },
                emergencyControl: {
                    name: '应急控制指令',
                    template: '对{target}执行{action}，影响范围{scope}，原因：{reason}',
                    parameters: ['target', 'action', 'scope', 'reason']
                },
                modeSwitch: {
                    name: '运行模式切换指令',
                    template: '将{target}切换至{mode}模式，预计持续{duration}',
                    parameters: ['target', 'mode', 'duration']
                }
            },
            // 指令优先级
            priorities: {
                emergency: 1,    // 紧急
                high: 2,        // 高优先级
                normal: 3,      // 普通
                low: 4          // 低优先级
            }
        };

        // 绑定事件处理器
        this.bindEventHandlers();
    }

    /**
     * 绑定事件处理器
     */
    bindEventHandlers() {
        // 快速调度按钮
        document.querySelectorAll('.quick-dispatch button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const type = e.target.dataset.dispatchType;
                this.showQuickDispatchModal(type);
            });
        });

        // 新建指令按钮
        document.getElementById('newCommandBtn')?.addEventListener('click', () => {
            this.showNewCommandModal();
        });
    }

    /**
     * 显示新建指令模态框
     */
    showNewCommandModal() {
        const modalHtml = `
            <div class="modal fade" id="newCommandModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header bg-dark text-white">
                            <h5 class="modal-title">新建调度指令</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <form id="commandForm">
                                <div class="row mb-3">
                                    <div class="col-md-6">
                                        <label class="form-label">指令类型</label>
                                        <select class="form-select" id="commandType" required>
                                            <option value="">选择类型...</option>
                                            <option value="powerAdjustment">功率调整指令</option>
                                            <option value="emergencyControl">应急控制指令</option>
                                            <option value="modeSwitch">运行模式切换指令</option>
                                        </select>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">优先级</label>
                                        <select class="form-select" id="commandPriority" required>
                                            <option value="emergency">紧急</option>
                                            <option value="high">高优先级</option>
                                            <option value="normal" selected>普通</option>
                                            <option value="low">低优先级</option>
                                        </select>
                                    </div>
                                </div>

                                <div class="row mb-3">
                                    <div class="col-md-6">
                                        <label class="form-label">目标设备/系统</label>
                                        <select class="form-select" id="commandTarget" required>
                                            <option value="">选择目标...</option>
                                            <optgroup label="发电系统">
                                                <option value="wind_all">全部风机</option>
                                                <option value="wind_group1">风机组#1</option>
                                                <option value="wind_group2">风机组#2</option>
                                                <option value="solar_all">全部光伏</option>
                                                <option value="solar_array1">光伏阵列#1</option>
                                                <option value="solar_array2">光伏阵列#2</option>
                                            </optgroup>
                                            <optgroup label="储能系统">
                                                <option value="storage_all">全部储能</option>
                                                <option value="storage_unit1">储能单元#1</option>
                                                <option value="storage_unit2">储能单元#2</option>
                                            </optgroup>
                                            <optgroup label="负荷系统">
                                                <option value="load_all">全部负荷</option>
                                                <option value="load_industrial">工业负荷</option>
                                                <option value="load_commercial">商业负荷</option>
                                                <option value="load_residential">居民负荷</option>
                                            </optgroup>
                                        </select>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">执行时间</label>
                                        <input type="datetime-local" class="form-control" id="commandTime" required>
                                    </div>
                                </div>

                                <div id="dynamicParameters">
                                    <!-- 动态参数区域 -->
                                </div>

                                <div class="mb-3">
                                    <label class="form-label">指令说明</label>
                                    <textarea class="form-control" id="commandDescription" rows="3" required></textarea>
                                </div>

                                <div class="mb-3">
                                    <label class="form-label">执行条件</label>
                                    <div class="conditions-container">
                                        <div class="form-check mb-2">
                                            <input class="form-check-input" type="checkbox" id="condition_weather">
                                            <label class="form-check-label">天气条件满足要求</label>
                                        </div>
                                        <div class="form-check mb-2">
                                            <input class="form-check-input" type="checkbox" id="condition_safety">
                                            <label class="form-check-label">安全措施到位</label>
                                        </div>
                                        <div class="form-check mb-2">
                                            <input class="form-check-input" type="checkbox" id="condition_system">
                                            <label class="form-check-label">系统状态正常</label>
                                        </div>
                                    </div>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer bg-dark text-white">
                            <button type="button" class="btn btn-light" data-bs-dismiss="modal">取消</button>
                            <button type="button" class="btn btn-primary" onclick="powerDispatchCommands.createCommand()">
                                创建指令
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // 添加模态框到页面
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('newCommandModal'));
        modal.show();

        // 监听指令类型变化
        document.getElementById('commandType')?.addEventListener('change', (e) => {
            this.updateDynamicParameters(e.target.value);
        });

        // 模态框关闭时移除
        document.getElementById('newCommandModal').addEventListener('hidden.bs.modal', function () {
            this.remove();
        });
    }

    /**
     * 更新动态参数
     */
    updateDynamicParameters(type) {
        const template = this.state.templates[type];
        if (!template) return;

        const container = document.getElementById('dynamicParameters');
        if (!container) return;

        let html = '';
        switch(type) {
            case 'powerAdjustment':
                html = `
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label class="form-label">调整动作</label>
                            <select class="form-select" id="param_action" required>
                                <option value="increase">增加</option>
                                <option value="decrease">减少</option>
                                <option value="set">设定</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">目标值</label>
                            <div class="input-group">
                                <input type="number" class="form-control" id="param_value" required>
                                <select class="form-select" id="param_unit" style="max-width: 80px;">
                                    <option value="MW">MW</option>
                                    <option value="%">%</option>
                                </select>
                            </div>
                        </div>
                    </div>
                `;
                break;
            case 'emergencyControl':
                html = `
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label class="form-label">控制动作</label>
                            <select class="form-select" id="param_action" required>
                                <option value="shutdown">紧急停机</option>
                                <option value="limit">限制出力</option>
                                <option value="isolate">系统隔离</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">影响范围</label>
                            <select class="form-select" id="param_scope" required>
                                <option value="single">单个设备</option>
                                <option value="group">设备组</option>
                                <option value="all">全部系统</option>
                            </select>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">原因说明</label>
                        <textarea class="form-control" id="param_reason" rows="2" required></textarea>
                    </div>
                `;
                break;
            case 'modeSwitch':
                html = `
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label class="form-label">目标模式</label>
                            <select class="form-select" id="param_mode" required>
                                <option value="normal">正常运行</option>
                                <option value="eco">经济运行</option>
                                <option value="peak">高峰应对</option>
                                <option value="maintenance">维护模式</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">预计持续时间</label>
                            <div class="input-group">
                                <input type="number" class="form-control" id="param_duration" required>
                                <span class="input-group-text">小时</span>
                            </div>
                        </div>
                    </div>
                `;
                break;
        }

        container.innerHTML = html;
    }

    /**
     * 创建调度指令
     */
    createCommand() {
        const form = document.getElementById('commandForm');
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        // 收集基本信息
        const commandData = {
            id: 'CMD' + Date.now(),
            type: document.getElementById('commandType').value,
            priority: document.getElementById('commandPriority').value,
            target: document.getElementById('commandTarget').value,
            executeTime: document.getElementById('commandTime').value,
            description: document.getElementById('commandDescription').value,
            conditions: {
                weather: document.getElementById('condition_weather').checked,
                safety: document.getElementById('condition_safety').checked,
                system: document.getElementById('condition_system').checked
            },
            parameters: this.collectCommandParameters(),
            status: 'pending',
            createdAt: new Date().toISOString()
        };

        // 保存指令
        this.saveCommand(commandData);

        // 关闭模态框
        bootstrap.Modal.getInstance(document.getElementById('newCommandModal')).hide();

        // 显示成功提示
        this.showToast('success', '调度指令创建成功');

        // 更新指令列表
        this.updateCommandsList();
    }

    /**
     * 收集指令参数
     */
    collectCommandParameters() {
        const type = document.getElementById('commandType').value;
        const params = {};

        switch(type) {
            case 'powerAdjustment':
                params.action = document.getElementById('param_action').value;
                params.value = document.getElementById('param_value').value;
                params.unit = document.getElementById('param_unit').value;
                break;
            case 'emergencyControl':
                params.action = document.getElementById('param_action').value;
                params.scope = document.getElementById('param_scope').value;
                params.reason = document.getElementById('param_reason').value;
                break;
            case 'modeSwitch':
                params.mode = document.getElementById('param_mode').value;
                params.duration = document.getElementById('param_duration').value;
                break;
        }

        return params;
    }

    /**
     * 保存调度指令
     */
    saveCommand(command) {
        this.state.commands.set(command.id, command);
        
        // 如果是紧急指令，立即执行
        if (command.priority === 'emergency') {
            this.executeCommand(command.id);
        }
    }

    /**
     * 执行调度指令
     */
    executeCommand(commandId) {
        const command = this.state.commands.get(commandId);
        if (!command) return;

        // 验证执行条件
        if (!this.validateExecutionConditions(command)) {
            this.showToast('warning', '执行条件不满足，请检查');
            return;
        }

        // 根据指令类型执行相应操作
        switch(command.type) {
            case 'powerAdjustment':
                this.executePowerAdjustment(command);
                break;
            case 'emergencyControl':
                this.executeEmergencyControl(command);
                break;
            case 'modeSwitch':
                this.executeModeSwitch(command);
                break;
        }

        // 更新指令状态
        command.status = 'executing';
        command.executeStartTime = new Date().toISOString();

        // 更新显示
        this.updateCommandsList();
    }

    /**
     * 验证执行条件
     */
    validateExecutionConditions(command) {
        // 检查天气条件
        if (command.conditions.weather && !this.checkWeatherCondition()) {
            return false;
        }

        // 检查安全措施
        if (command.conditions.safety && !this.checkSafetyMeasures()) {
            return false;
        }

        // 检查系统状态
        if (command.conditions.system && !this.checkSystemStatus()) {
            return false;
        }

        return true;
    }

    /**
     * 执行功率调整
     */
    executePowerAdjustment(command) {
        const { target, parameters } = command;
        const value = parseFloat(parameters.value);
        
        if (window.powerFlowManagement) {
            // 调用电力流程管理的相关方法
            if (target.startsWith('wind_')) {
                window.powerFlowManagement.adjustGenerationOutput('wind', value);
            } else if (target.startsWith('solar_')) {
                window.powerFlowManagement.adjustGenerationOutput('solar', value);
            }
        }
    }

    /**
     * 执行应急控制
     */
    executeEmergencyControl(command) {
        const { target, parameters } = command;
        
        if (window.powerFlowManagement) {
            switch(parameters.action) {
                case 'shutdown':
                    window.powerFlowManagement.emergencyShutdown(target);
                    break;
                case 'limit':
                    window.powerFlowManagement.setLoadLimit(target);
                    break;
                case 'isolate':
                    window.powerFlowManagement.isolateSystem(target);
                    break;
            }
        }
    }

    /**
     * 执行模式切换
     */
    executeModeSwitch(command) {
        const { target, parameters } = command;
        
        if (window.powerFlowManagement) {
            window.powerFlowManagement.switchOperationMode(target, parameters.mode);
        }
    }

    /**
     * 更新指令列表
     */
    updateCommandsList() {
        const container = document.querySelector('.recent-commands');
        if (!container) return;

        // 按时间排序
        const sortedCommands = Array.from(this.state.commands.values())
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

        container.innerHTML = sortedCommands.map(command => this.generateCommandHtml(command)).join('');
    }

    /**
     * 生成指令HTML
     */
    generateCommandHtml(command) {
        const template = this.state.templates[command.type];
        const priorityColors = {
            emergency: 'danger',
            high: 'warning',
            normal: 'info',
            low: 'secondary'
        };

        return `
            <div class="d-flex align-items-center mb-3">
                <div class="flex-shrink-0">
                    <span class="badge bg-soft-${priorityColors[command.priority]} p-2">
                        <i class="fas fa-bolt"></i>
                    </span>
                </div>
                <div class="flex-grow-1 ms-3">
                    <h6 class="mb-1">${template.name}</h6>
                    <p class="text-muted mb-0">${this.formatCommandContent(command)}</p>
                </div>
                <div class="flex-shrink-0">
                    <div class="btn-group">
                        <button class="btn btn-sm btn-${command.status === 'pending' ? 'primary' : 'light'}"
                                onclick="powerDispatchCommands.executeCommand('${command.id}')"
                                ${command.status !== 'pending' ? 'disabled' : ''}>
                            <i class="fas fa-play"></i>
                        </button>
                        <button class="btn btn-sm btn-danger"
                                onclick="powerDispatchCommands.cancelCommand('${command.id}')"
                                ${command.status === 'completed' ? 'disabled' : ''}>
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * 格式化指令内容
     */
    formatCommandContent(command) {
        const template = this.state.templates[command.type];
        if (!template) return '';

        let content = template.template;
        const params = {
            target: this.getTargetName(command.target),
            time: new Date(command.executeTime).toLocaleString(),
            ...command.parameters
        };

        // 替换模板中的参数
        for (const [key, value] of Object.entries(params)) {
            content = content.replace(`{${key}}`, value);
        }

        return content;
    }

    /**
     * 获取目标名称
     */
    getTargetName(target) {
        const targetNames = {
            'wind_all': '全部风机',
            'wind_group1': '风机组#1',
            'wind_group2': '风机组#2',
            'solar_all': '全部光伏',
            'solar_array1': '光伏阵列#1',
            'solar_array2': '光伏阵列#2',
            'storage_all': '全部储能',
            'storage_unit1': '储能单元#1',
            'storage_unit2': '储能单元#2',
            'load_all': '全部负荷',
            'load_industrial': '工业负荷',
            'load_commercial': '商业负荷',
            'load_residential': '居民负荷'
        };
        return targetNames[target] || target;
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

    /**
     * 显示快速调度模态框
     */
    showQuickDispatchModal(type) {
        let title, content;
        switch(type) {
            case 'wind':
                title = '风电快速调节';
                content = this.getWindControlContent();
                break;
            case 'solar':
                title = '光伏快速调节';
                content = this.getSolarControlContent();
                break;
            case 'storage':
                title = '储能快速控制';
                content = this.getStorageControlContent();
                break;
            case 'load':
                title = '负荷快速管理';
                content = this.getLoadControlContent();
                break;
        }

        const modalHtml = `
            <div class="modal fade" id="quickDispatchModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header bg-dark text-white">
                            <h5 class="modal-title">${title}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            ${content}
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('quickDispatchModal'));
        modal.show();

        // 模态框关闭时移除
        document.getElementById('quickDispatchModal').addEventListener('hidden.bs.modal', function () {
            this.remove();
        });
    }

    /**
     * 获取风电控制内容
     */
    getWindControlContent() {
        return `
            <div class="quick-control-panel">
                <div class="current-status mb-4">
                    <div class="row">
                        <div class="col-6">
                            <label class="form-label">当前功率</label>
                            <h3 class="mb-0">320.5 <small>MW</small></h3>
                        </div>
                        <div class="col-6">
                            <label class="form-label">可用容量</label>
                            <h3 class="mb-0">450.0 <small>MW</small></h3>
                        </div>
                    </div>
                </div>

                <div class="control-actions">
                    <div class="mb-3">
                        <label class="form-label">调整方式</label>
                        <div class="btn-group w-100">
                            <button class="btn btn-outline-primary active" data-action="power">功率</button>
                            <button class="btn btn-outline-primary" data-action="percent">百分比</button>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">快速调整</label>
                        <div class="btn-group w-100">
                            <button class="btn btn-primary" onclick="powerDispatchCommands.quickAdjust('wind', -10)">-10</button>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.quickAdjust('wind', -5)">-5</button>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.quickAdjust('wind', 5)">+5</button>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.quickAdjust('wind', 10)">+10</button>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">精确调整</label>
                        <div class="input-group">
                            <input type="number" class="form-control" id="windPowerValue" placeholder="输入目标值">
                            <span class="input-group-text">MW</span>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.preciseAdjust('wind')">
                                确定
                            </button>
                        </div>
                    </div>
                </div>

                <div class="optimization-actions mt-4">
                    <button class="btn btn-success w-100 mb-2" onclick="powerDispatchCommands.optimizeGeneration('wind')">
                        <i class="fas fa-magic me-1"></i>优化出力
                    </button>
                    <button class="btn btn-info w-100" onclick="powerDispatchCommands.switchMode('wind', 'eco')">
                        <i class="fas fa-leaf me-1"></i>切换至经济模式
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * 获取光伏控制内容
     */
    getSolarControlContent() {
        return `
            <div class="quick-control-panel">
                <div class="current-status mb-4">
                    <div class="row">
                        <div class="col-6">
                            <label class="form-label">当前功率</label>
                            <h3 class="mb-0">425.8 <small>MW</small></h3>
                        </div>
                        <div class="col-6">
                            <label class="form-label">可用容量</label>
                            <h3 class="mb-0">520.0 <small>MW</small></h3>
                        </div>
                    </div>
                </div>

                <div class="control-actions">
                    <div class="mb-3">
                        <label class="form-label">调整方式</label>
                        <div class="btn-group w-100">
                            <button class="btn btn-outline-primary active" data-action="power">功率</button>
                            <button class="btn btn-outline-primary" data-action="percent">百分比</button>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">快速调整</label>
                        <div class="btn-group w-100">
                            <button class="btn btn-primary" onclick="powerDispatchCommands.quickAdjust('solar', -10)">-10</button>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.quickAdjust('solar', -5)">-5</button>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.quickAdjust('solar', 5)">+5</button>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.quickAdjust('solar', 10)">+10</button>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">精确调整</label>
                        <div class="input-group">
                            <input type="number" class="form-control" id="solarPowerValue" placeholder="输入目标值">
                            <span class="input-group-text">MW</span>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.preciseAdjust('solar')">
                                确定
                            </button>
                        </div>
                    </div>
                </div>

                <div class="optimization-actions mt-4">
                    <button class="btn btn-success w-100 mb-2" onclick="powerDispatchCommands.optimizeGeneration('solar')">
                        <i class="fas fa-magic me-1"></i>优化出力
                    </button>
                    <button class="btn btn-info w-100" onclick="powerDispatchCommands.switchMode('solar', 'eco')">
                        <i class="fas fa-leaf me-1"></i>切换至经济模式
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * 获取储能控制内容
     */
    getStorageControlContent() {
        return `
            <div class="quick-control-panel">
                <div class="current-status mb-4">
                    <div class="row">
                        <div class="col-6">
                            <label class="form-label">当前储能</label>
                            <h3 class="mb-0">110.0 <small>MWh</small></h3>
                        </div>
                        <div class="col-6">
                            <label class="form-label">SOC</label>
                            <h3 class="mb-0">55 <small>%</small></h3>
                        </div>
                    </div>
                </div>

                <div class="control-actions">
                    <div class="mb-3">
                        <label class="form-label">运行模式</label>
                        <div class="btn-group w-100">
                            <button class="btn btn-outline-primary active" onclick="powerDispatchCommands.setStorageMode('charge')">充电</button>
                            <button class="btn btn-outline-primary" onclick="powerDispatchCommands.setStorageMode('discharge')">放电</button>
                            <button class="btn btn-outline-primary" onclick="powerDispatchCommands.setStorageMode('standby')">待机</button>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">功率调整</label>
                        <div class="input-group">
                            <input type="number" class="form-control" id="storagePowerValue" placeholder="输入功率值">
                            <span class="input-group-text">MW</span>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.adjustStoragePower()">
                                确定
                            </button>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">目标SOC</label>
                        <div class="input-group">
                            <input type="number" class="form-control" id="targetSoc" placeholder="输入目标SOC">
                            <span class="input-group-text">%</span>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.setTargetSoc()">
                                确定
                            </button>
                        </div>
                    </div>
                </div>

                <div class="optimization-actions mt-4">
                    <button class="btn btn-success w-100 mb-2" onclick="powerDispatchCommands.optimizeStorage()">
                        <i class="fas fa-magic me-1"></i>优化策略
                    </button>
                    <button class="btn btn-info w-100" onclick="powerDispatchCommands.switchStorageMode('eco')">
                        <i class="fas fa-leaf me-1"></i>经济模式
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * 获取负荷控制内容
     */
    getLoadControlContent() {
        return `
            <div class="quick-control-panel">
                <div class="current-status mb-4">
                    <div class="row">
                        <div class="col-6">
                            <label class="form-label">当前负荷</label>
                            <h3 class="mb-0">856.3 <small>MW</small></h3>
                        </div>
                        <div class="col-6">
                            <label class="form-label">峰值负荷</label>
                            <h3 class="mb-0">1000.0 <small>MW</small></h3>
                        </div>
                    </div>
                </div>

                <div class="control-actions">
                    <div class="mb-3">
                        <label class="form-label">负荷类型</label>
                        <select class="form-select" id="loadType">
                            <option value="all">全部负荷</option>
                            <option value="industrial">工业负荷</option>
                            <option value="commercial">商业负荷</option>
                            <option value="residential">居民负荷</option>
                        </select>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">控制动作</label>
                        <div class="btn-group w-100">
                            <button class="btn btn-warning" onclick="powerDispatchCommands.loadControl('shed')">切除</button>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.loadControl('limit')">限制</button>
                            <button class="btn btn-success" onclick="powerDispatchCommands.loadControl('restore')">恢复</button>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">限制值</label>
                        <div class="input-group">
                            <input type="number" class="form-control" id="loadLimitValue" placeholder="输入限制值">
                            <span class="input-group-text">MW</span>
                            <button class="btn btn-primary" onclick="powerDispatchCommands.setLoadLimit()">
                                确定
                            </button>
                        </div>
                    </div>
                </div>

                <div class="optimization-actions mt-4">
                    <button class="btn btn-success w-100 mb-2" onclick="powerDispatchCommands.optimizeLoad()">
                        <i class="fas fa-magic me-1"></i>优化负荷
                    </button>
                    <button class="btn btn-info w-100" onclick="powerDispatchCommands.switchLoadMode('valley')">
                        <i class="fas fa-clock me-1"></i>错峰模式
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * 快速调整功率
     */
    quickAdjust(type, delta) {
        const command = {
            type: 'powerAdjustment',
            priority: 'normal',
            target: `${type}_all`,
            parameters: {
                action: delta > 0 ? 'increase' : 'decrease',
                value: Math.abs(delta),
                unit: 'MW'
            },
            executeTime: new Date().toISOString(),
            description: `${type === 'wind' ? '风电' : '光伏'}功率${delta > 0 ? '增加' : '减少'}${Math.abs(delta)}MW`,
            conditions: {
                weather: false,
                safety: true,
                system: true
            }
        };

        this.saveCommand(command);
        this.executeCommand(command.id);
        bootstrap.Modal.getInstance(document.getElementById('quickDispatchModal')).hide();
    }

    /**
     * 精确调整功率
     */
    preciseAdjust(type) {
        const value = parseFloat(document.getElementById(`${type}PowerValue`).value);
        if (isNaN(value)) {
            this.showToast('warning', '请输入有效的功率值');
            return;
        }

        const command = {
            type: 'powerAdjustment',
            priority: 'normal',
            target: `${type}_all`,
            parameters: {
                action: 'set',
                value: value,
                unit: 'MW'
            },
            executeTime: new Date().toISOString(),
            description: `${type === 'wind' ? '风电' : '光伏'}功率设定为${value}MW`,
            conditions: {
                weather: false,
                safety: true,
                system: true
            }
        };

        this.saveCommand(command);
        this.executeCommand(command.id);
        bootstrap.Modal.getInstance(document.getElementById('quickDispatchModal')).hide();
    }

    /**
     * 执行应急指令
     */
    executeEmergencyCommand(action) {
        const command = {
            type: 'emergencyControl',
            priority: 'emergency',
            target: 'all',
            parameters: {
                action: action,
                scope: 'all',
                reason: '应急控制'
            },
            executeTime: new Date().toISOString(),
            description: '执行应急控制指令',
            conditions: {
                weather: false,
                safety: true,
                system: true
            }
        };

        this.saveCommand(command);
        bootstrap.Modal.getInstance(document.getElementById('emergencyControlModal')).hide();
    }

    /**
     * 检查天气条件
     */
    checkWeatherCondition() {
        // 这里应该调用天气服务获取实时天气数据
        // 暂时返回true
        return true;
    }

    /**
     * 检查安全措施
     */
    checkSafetyMeasures() {
        // 这里应该检查各项安全措施是否到位
        // 暂时返回true
        return true;
    }

    /**
     * 检查系统状态
     */
    checkSystemStatus() {
        // 这里应该检查系统各项指标是否正常
        // 暂时返回true
        return true;
    }

    /**
     * 取消指令
     */
    cancelCommand(commandId) {
        const command = this.state.commands.get(commandId);
        if (!command) return;

        if (command.status === 'executing') {
            // 如果指令正在执行，需要执行回滚操作
            this.rollbackCommand(command);
        }

        command.status = 'cancelled';
        this.updateCommandsList();
        this.showToast('info', '指令已取消');
    }

    /**
     * 回滚指令
     */
    rollbackCommand(command) {
        switch(command.type) {
            case 'powerAdjustment':
                // 回滚功率调整
                if (window.powerFlowManagement) {
                    window.powerFlowManagement.rollbackAdjustment(command.target);
                }
                break;
            case 'modeSwitch':
                // 回滚模式切换
                if (window.powerFlowManagement) {
                    window.powerFlowManagement.rollbackModeSwitch(command.target);
                }
                break;
        }
    }

    /**
     * 导出指令历史
     */
    exportHistory() {
        const history = Array.from(this.state.commands.values())
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

        // 准备导出数据
        const exportData = history.map(command => ({
            '指令ID': command.id,
            '指令类型': this.state.templates[command.type].name,
            '优先级': this.getPriorityName(command.priority),
            '目标': this.getTargetName(command.target),
            '执行时间': new Date(command.executeTime).toLocaleString(),
            '状态': this.getStatusName(command.status),
            '描述': command.description
        }));

        // 创建工作簿
        const wb = XLSX.utils.book_new();
        const ws = XLSX.utils.json_to_sheet(exportData);
        XLSX.utils.book_append_sheet(wb, ws, '调度指令历史');

        // 导出文件
        XLSX.writeFile(wb, `调度指令历史_${new Date().toISOString().split('T')[0]}.xlsx`);
    }

    /**
     * 清空指令历史
     */
    clearHistory() {
        if (confirm('确定要清空历史记录吗？此操作不可恢复。')) {
            // 保留正在执行的指令
            const executingCommands = Array.from(this.state.commands.values())
                .filter(command => command.status === 'executing');
            
            this.state.commands.clear();
            
            // 重新添加正在执行的指令
            executingCommands.forEach(command => {
                this.state.commands.set(command.id, command);
            });

            this.updateCommandsList();
            this.showToast('success', '历史记录已清空');
        }
    }

    /**
     * 获取优先级名称
     */
    getPriorityName(priority) {
        const priorityNames = {
            emergency: '紧急',
            high: '高',
            normal: '普通',
            low: '低'
        };
        return priorityNames[priority] || priority;
    }

    /**
     * 获取状态名称
     */
    getStatusName(status) {
        const statusNames = {
            pending: '待执行',
            executing: '执行中',
            completed: '已完成',
            cancelled: '已取消',
            failed: '执行失败'
        };
        return statusNames[status] || status;
    }

    /**
     * 初始化调度效果图表
     */
    initDispatchEffectChart() {
        const chartDom = document.querySelector('.dispatch-effect-chart');
        if (!chartDom) return;

        const chart = echarts.init(chartDom);
        const option = {
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
                axisLine: {
                    lineStyle: {
                        color: '#ccc'
                    }
                },
                axisLabel: {
                    color: '#ccc'
                }
            },
            yAxis: {
                type: 'value',
                name: '功率(MW)',
                nameTextStyle: {
                    color: '#ccc'
                },
                axisLine: {
                    lineStyle: {
                        color: '#ccc'
                    }
                },
                axisLabel: {
                    color: '#ccc'
                },
                splitLine: {
                    lineStyle: {
                        color: 'rgba(204, 204, 204, 0.1)'
                    }
                }
            },
            series: [
                {
                    name: '计划功率',
                    type: 'line',
                    data: [320, 332, 301, 334, 390, 330, 320, 310],
                    smooth: true,
                    lineStyle: {
                        width: 2,
                        type: 'dashed'
                    },
                    itemStyle: {
                        color: '#2ab57d'
                    }
                },
                {
                    name: '实际功率',
                    type: 'line',
                    data: [320, 332, 301, 334, 390, 330, 320, 310].map(v => v + Math.random() * 20 - 10),
                    smooth: true,
                    lineStyle: {
                        width: 2
                    },
                    itemStyle: {
                        color: '#4F8CBE'
                    }
                }
            ]
        };

        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());
    }

    /**
     * 确认所有预警
     */
    acknowledgeAllAlerts() {
        document.querySelectorAll('.dispatch-alerts .alert').forEach(alert => {
            alert.remove();
        });
        this.showToast('success', '已确认所有预警');
    }

    /**
     * 确认单个预警
     */
    acknowledgeAlert(alertId) {
        const alert = document.querySelector(`[data-alert-id="${alertId}"]`);
        if (alert) {
            alert.remove();
            this.showToast('success', '已确认预警');
        }
    }

    /**
     * 更新调度效果统计
     */
    updateDispatchStats() {
        const stats = this.calculateDispatchStats();
        
        // 更新DOM
        document.getElementById('executionRate').textContent = `${stats.executionRate}%`;
        document.getElementById('responseTime').textContent = `${stats.responseTime}s`;
        document.getElementById('accuracy').textContent = `${stats.accuracy}%`;

        // 更新图表
        this.updateDispatchEffectChart(stats);
    }

    /**
     * 计算调度效果统计
     */
    calculateDispatchStats() {
        const commands = Array.from(this.state.commands.values());
        const completedCommands = commands.filter(cmd => cmd.status === 'completed');
        
        // 计算执行率
        const executionRate = (completedCommands.length / commands.length * 100).toFixed(1);
        
        // 计算平均响应时间
        const responseTimes = completedCommands
            .map(cmd => new Date(cmd.executeStartTime) - new Date(cmd.executeTime))
            .filter(time => time > 0);
        const avgResponseTime = (responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length / 1000).toFixed(1);
        
        // 计算准确率（实际值与目标值的偏差）
        const accuracies = completedCommands
            .filter(cmd => cmd.type === 'powerAdjustment')
            .map(cmd => {
                const target = parseFloat(cmd.parameters.value);
                const actual = cmd.actualValue || target; // 如果没有实际值，假设达到了目标
                return (1 - Math.abs(actual - target) / target) * 100;
            });
        const accuracy = (accuracies.reduce((a, b) => a + b, 0) / accuracies.length).toFixed(1);

        return {
            executionRate,
            responseTime: avgResponseTime,
            accuracy
        };
    }

    /**
     * 更新调度效果图表
     */
    updateDispatchEffectChart(stats) {
        const chart = echarts.getInstanceByDom(document.querySelector('.dispatch-effect-chart'));
        if (!chart) return;

        // 生成新的数据
        const now = new Date();
        const data = Array.from({length: 8}, (_, i) => {
            const time = new Date(now.getTime() - (7 - i) * 3600 * 1000);
            return {
                time: time.toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit'}),
                planned: Math.round(300 + Math.random() * 100),
                actual: Math.round(300 + Math.random() * 100)
            };
        });

        chart.setOption({
            xAxis: {
                data: data.map(d => d.time)
            },
            series: [
                {
                    name: '计划功率',
                    data: data.map(d => d.planned)
                },
                {
                    name: '实际功率',
                    data: data.map(d => d.actual)
                }
            ]
        });
    }
}

// 初始化调度指令管理
document.addEventListener('DOMContentLoaded', () => {
    window.powerDispatchCommands = new PowerDispatchCommands();
}); 