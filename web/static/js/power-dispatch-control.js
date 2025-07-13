class PowerDispatchControl {
    constructor() {
        this.initializeEventListeners();
        this.initializeValues();
        this.initializeModal();
    }

    // 初始化事件监听器
    initializeEventListeners() {
        // 发电控制按钮事件
        document.querySelectorAll('[data-generation-control]').forEach(button => {
            button.addEventListener('click', (e) => {
                const type = e.target.closest('[data-generation-control]').dataset.generationType;
                const action = e.target.closest('[data-generation-control]').dataset.generationAction;
                this.handleGenerationControl(type, action);
            });
        });

        // 储能控制按钮事件
        document.querySelectorAll('[data-storage-control]').forEach(button => {
            button.addEventListener('click', (e) => {
                const action = e.target.closest('[data-storage-control]').dataset.storageAction;
                this.handleStorageControl(action);
            });
        });

        // 负荷控制按钮事件
        document.querySelectorAll('[data-load-control]').forEach(button => {
            button.addEventListener('click', (e) => {
                const action = e.target.closest('[data-load-control]').dataset.loadAction;
                this.handleLoadControl(action);
            });
        });
    }

    // 初始化数值
    initializeValues() {
        this.values = {
            wind: {
                current: 320.5,
                max: 450,
                min: 0,
                step: 10
            },
            solar: {
                current: 425.8,
                max: 520,
                min: 0,
                step: 10
            },
            storage: {
                current: 110.0,
                max: 200,
                min: 0,
                step: 5
            },
            load: {
                current: 856.3,
                max: 1000,
                min: 0,
                step: 20
            }
        };

        // 更新显示值
        this.updateDisplayValues();
    }

    // 处理发电控制
    handleGenerationControl(type, action) {
        const data = this.values[type];
        let newValue = data.current;

        switch (action) {
            case 'increase':
                newValue = Math.min(data.current + data.step, data.max);
                break;
            case 'decrease':
                newValue = Math.max(data.current - data.step, data.min);
                break;
            case 'optimize':
                this.showOptimizationModal(type);
                return;
            case 'emergency':
                this.handleEmergencyShutdown(type);
                return;
        }

        // 更新值并显示
        this.values[type].current = newValue;
        this.updateDisplayValues();
        this.updateProgressBars();
        this.notifyChange(type, action, newValue);
    }

    // 处理储能控制
    handleStorageControl(action) {
        switch (action) {
            case 'charge':
                if (this.values.storage.current < this.values.storage.max) {
                    this.values.storage.current = Math.min(
                        this.values.storage.current + this.values.storage.step,
                        this.values.storage.max
                    );
                }
                break;
            case 'discharge':
                if (this.values.storage.current > this.values.storage.min) {
                    this.values.storage.current = Math.max(
                        this.values.storage.current - this.values.storage.step,
                        this.values.storage.min
                    );
                }
                break;
            case 'optimize':
                this.showStorageOptimizationModal();
                return;
            case 'standby':
                this.setStorageStandbyMode();
                return;
        }

        this.updateDisplayValues();
        this.updateProgressBars();
        this.notifyChange('storage', action, this.values.storage.current);
    }

    // 处理负荷控制
    handleLoadControl(action) {
        switch (action) {
            case 'shed':
                this.showLoadSheddingModal();
                break;
            case 'limit':
                this.showLoadLimitModal();
                break;
            case 'optimize':
                this.showLoadOptimizationModal();
                break;
            case 'restore':
                this.restoreLoad();
                break;
        }
    }

    // 更新显示值
    updateDisplayValues() {
        // 更新发电值显示
        document.querySelector('[data-generation-type="wind"][data-power-value]').textContent = 
            this.values.wind.current.toFixed(1);
        document.querySelector('[data-generation-type="solar"][data-power-value]').textContent = 
            this.values.solar.current.toFixed(1);
        document.querySelector('[data-storage-value]').textContent = 
            this.values.storage.current.toFixed(1);
        document.querySelector('[data-load-value]').textContent = 
            this.values.load.current.toFixed(1);
    }

    // 更新进度条
    updateProgressBars() {
        // 更新各类型的进度条
        const updateProgress = (type, value, max) => {
            const progressBar = document.querySelector(`[data-generation-type="${type}"]`)
                ?.closest('.card')
                ?.querySelector('.progress-bar');
            if (progressBar) {
                const percentage = (value / max * 100).toFixed(0);
                progressBar.style.width = `${percentage}%`;
                progressBar.closest('.card').querySelector('small:first-of-type').textContent = 
                    `利用率: ${percentage}%`;
            }
        };

        updateProgress('wind', this.values.wind.current, this.values.wind.max);
        updateProgress('solar', this.values.solar.current, this.values.solar.max);

        // 更新储能进度条
        const storageProgress = document.querySelector('[data-storage-value]')
            ?.closest('.card')
            ?.querySelector('.progress-bar');
        if (storageProgress) {
            const storagePercentage = (this.values.storage.current / this.values.storage.max * 100).toFixed(0);
            storageProgress.style.width = `${storagePercentage}%`;
            storageProgress.closest('.card').querySelector('small:first-of-type').textContent = 
                `SOC: ${storagePercentage}%`;
        }

        // 更新负荷进度条
        const loadProgress = document.querySelector('[data-load-value]')
            ?.closest('.card')
            ?.querySelector('.progress-bar');
        if (loadProgress) {
            const loadPercentage = (this.values.load.current / this.values.load.max * 100).toFixed(0);
            loadProgress.style.width = `${loadPercentage}%`;
            loadProgress.closest('.card').querySelector('small:first-of-type').textContent = 
                `负载率: ${loadPercentage}%`;
        }
    }

    // 显示优化模态框
    showOptimizationModal(type) {
        const title = {
            wind: '风电优化配置',
            solar: '光伏优化配置'
        }[type];

        // 使用Bootstrap模态框
        const modal = new bootstrap.Modal(document.getElementById('dispatchModal'));
        document.querySelector('#dispatchModal .modal-title').textContent = title;
        modal.show();
    }

    // 处理紧急停机
    handleEmergencyShutdown(type) {
        // 创建自定义确认对话框
        const confirmDialog = document.createElement('div');
        confirmDialog.className = 'custom-dialog';
        confirmDialog.innerHTML = `
            <div class="custom-dialog-content">
                <div class="dialog-header">
                    <h5>确认操作</h5>
                </div>
                <div class="dialog-body">
                    <p>确认要执行${type === 'wind' ? '风机组' : '光伏组件'}紧急停机吗？</p>
                </div>
                <div class="dialog-footer">
                    <button class="btn btn-secondary" data-action="cancel">取消</button>
                    <button class="btn btn-danger" data-action="confirm">确认</button>
                </div>
            </div>
        `;

        // 添加到页面
        document.body.appendChild(confirmDialog);

        // 添加动画效果
        setTimeout(() => confirmDialog.classList.add('show'), 10);

        // 处理按钮点击
        confirmDialog.addEventListener('click', (e) => {
            const action = e.target.getAttribute('data-action');
            if (action === 'confirm') {
                this.values[type].current = 0;
                this.updateDisplayValues();
                this.updateProgressBars();
                this.notifyChange(type, 'emergency', 0);
                
                // 显示紧急停机通知
                this.showCustomNotification('warning', `${type === 'wind' ? '风机组' : '光伏组件'}已紧急停机`);
            }

            // 移除对话框
            confirmDialog.classList.remove('show');
            setTimeout(() => confirmDialog.remove(), 300);
        });
    }

    // 显示自定义通知
    showCustomNotification(type, message) {
        const notification = document.createElement('div');
        notification.className = `custom-notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="ri-${this.getNotificationIcon(type)} me-2"></i>
                <span>${message}</span>
            </div>
        `;

        // 添加到页面
        const notificationContainer = document.querySelector('.notification-container') || (() => {
            const container = document.createElement('div');
            container.className = 'notification-container';
            document.body.appendChild(container);
            return container;
        })();

        notificationContainer.appendChild(notification);

        // 添加动画效果
        setTimeout(() => notification.classList.add('show'), 10);

        // 自动移除
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // 获取通知图标
    getNotificationIcon(type) {
        const icons = {
            success: 'check-line',
            warning: 'alert-line',
            danger: 'error-warning-line',
            info: 'information-line'
        };
        return icons[type] || 'notification-line';
    }

    // 显示储能优化模态框
    showStorageOptimizationModal() {
        const modal = new bootstrap.Modal(document.getElementById('dispatchModal'));
        document.querySelector('#dispatchModal .modal-title').textContent = '储能优化配置';
        modal.show();
    }

    // 设置储能待机模式
    setStorageStandbyMode() {
        // 创建自定义确认对话框
        const confirmDialog = document.createElement('div');
        confirmDialog.className = 'custom-dialog';
        confirmDialog.innerHTML = `
            <div class="custom-dialog-content">
                <div class="dialog-header">
                    <h5>确认操作</h5>
                </div>
                <div class="dialog-body">
                    <p>确认要将储能系统切换到待机模式吗？</p>
                </div>
                <div class="dialog-footer">
                    <button class="btn btn-secondary" data-action="cancel">取消</button>
                    <button class="btn btn-primary" data-action="confirm">确认</button>
                </div>
            </div>
        `;

        document.body.appendChild(confirmDialog);
        setTimeout(() => confirmDialog.classList.add('show'), 10);

        confirmDialog.addEventListener('click', (e) => {
            const action = e.target.getAttribute('data-action');
            if (action === 'confirm') {
                this.showCustomNotification('info', '储能系统已切换到待机模式');
            }
            confirmDialog.classList.remove('show');
            setTimeout(() => confirmDialog.remove(), 300);
        });
    }

    // 显示负荷切除模态框
    showLoadSheddingModal() {
        const modal = new bootstrap.Modal(document.getElementById('dispatchModal'));
        document.querySelector('#dispatchModal .modal-title').textContent = '负荷切除配置';
        modal.show();
    }

    // 显示负荷限制模态框
    showLoadLimitModal() {
        const modal = new bootstrap.Modal(document.getElementById('dispatchModal'));
        document.querySelector('#dispatchModal .modal-title').textContent = '负荷限制配置';
        modal.show();
    }

    // 显示负荷优化模态框
    showLoadOptimizationModal() {
        const modal = new bootstrap.Modal(document.getElementById('dispatchModal'));
        document.querySelector('#dispatchModal .modal-title').textContent = '负荷优化配置';
        modal.show();
    }

    // 恢复负荷
    restoreLoad() {
        // 创建自定义确认对话框
        const confirmDialog = document.createElement('div');
        confirmDialog.className = 'custom-dialog';
        confirmDialog.innerHTML = `
            <div class="custom-dialog-content">
                <div class="dialog-header">
                    <h5>确认操作</h5>
                </div>
                <div class="dialog-body">
                    <p>确认要恢复负荷吗？</p>
                </div>
                <div class="dialog-footer">
                    <button class="btn btn-secondary" data-action="cancel">取消</button>
                    <button class="btn btn-primary" data-action="confirm">确认</button>
                </div>
            </div>
        `;

        document.body.appendChild(confirmDialog);
        setTimeout(() => confirmDialog.classList.add('show'), 10);

        confirmDialog.addEventListener('click', (e) => {
            const action = e.target.getAttribute('data-action');
            if (action === 'confirm') {
                this.showCustomNotification('success', '负荷已恢复正常');
            }
            confirmDialog.classList.remove('show');
            setTimeout(() => confirmDialog.remove(), 300);
        });
    }

    // 通知变更
    notifyChange(type, action, value) {
        // 记录到调度日志
        this.addToDispatchLog(type, action, value);
        
        // 如果是重要变更，更新系统状态
        this.updateSystemStatus(type, value);
    }

    // 添加到调度日志
    addToDispatchLog(type, action, value) {
        const logContainer = document.querySelector('.log-container');
        if (!logContainer) return;

        const now = new Date();
        const timeString = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;

        const logItem = document.createElement('div');
        logItem.className = 'log-item';
        logItem.innerHTML = `
            <small class="text-muted">${timeString}</small>
            <span class="text-${this.getActionColor(action)}">
                ${this.getActionDescription(type, action, value)}
            </span>
        `;

        logContainer.insertBefore(logItem, logContainer.firstChild);
    }

    // 获取操作描述
    getActionDescription(type, action, value) {
        const typeNames = {
            wind: '风电',
            solar: '光伏',
            storage: '储能',
            load: '负荷'
        };

        const actionDescriptions = {
            increase: '增加',
            decrease: '减少',
            optimize: '优化',
            emergency: '紧急停机',
            charge: '充电',
            discharge: '放电',
            standby: '待机',
            shed: '切除',
            limit: '限制',
            restore: '恢复'
        };

        return `${typeNames[type]}${actionDescriptions[action]}${value !== undefined ? ` (${value.toFixed(1)} MW)` : ''}`;
    }

    // 获取操作颜色
    getActionColor(action) {
        const actionColors = {
            increase: 'success',
            decrease: 'warning',
            optimize: 'info',
            emergency: 'danger',
            charge: 'success',
            discharge: 'warning',
            standby: 'secondary',
            shed: 'danger',
            limit: 'warning',
            restore: 'success'
        };

        return actionColors[action] || 'primary';
    }

    // 更新系统状态
    updateSystemStatus(type, value) {
        // 更新系统频率
        const frequencySpan = document.querySelector('[data-grid-frequency]');
        if (frequencySpan) {
            const baseFrequency = 50.00;
            const variation = (Math.random() * 0.04 - 0.02).toFixed(2);
            frequencySpan.textContent = (baseFrequency + parseFloat(variation)).toFixed(2);
        }

        // 更新功率因数
        const powerFactorSpan = document.querySelector('[data-power-factor]');
        if (powerFactorSpan) {
            const basePowerFactor = 0.95;
            const variation = (Math.random() * 0.02 - 0.01).toFixed(2);
            powerFactorSpan.textContent = (basePowerFactor + parseFloat(variation)).toFixed(2);
        }

        // 更新线损率
        const lineLossSpan = document.querySelector('[data-line-loss]');
        if (lineLossSpan) {
            const baseLineLoss = 12.5;
            const variation = (Math.random() * 0.4 - 0.2).toFixed(1);
            lineLossSpan.textContent = (baseLineLoss + parseFloat(variation)).toFixed(1);
        }
    }

    // 初始化模态框
    initializeModal() {
        // 获取模态框元素
        const modal = document.getElementById('dispatchModal');
        if (!modal) return;

        // 监听调度类型变化
        const dispatchType = modal.querySelector('#dispatchType');
        if (dispatchType) {
            dispatchType.addEventListener('change', () => this.handleDispatchTypeChange(dispatchType.value));
        }

        // 监听确认按钮点击
        const confirmButton = modal.querySelector('#confirmDispatch');
        if (confirmButton) {
            confirmButton.addEventListener('click', () => this.handleDispatchConfirm());
        }

        // 初始化设备选择器
        this.initializeDeviceSelector();

        // 初始化日期时间选择器
        const executeTime = modal.querySelector('#executeTime');
        if (executeTime) {
            const now = new Date();
            now.setMinutes(now.getMinutes() + 5); // 默认设置为5分钟后
            executeTime.value = now.toISOString().slice(0, 16);
        }
    }

    // 初始化设备选择器
    initializeDeviceSelector() {
        const deviceSelect = document.querySelector('#targetDevice');
        if (!deviceSelect) return;

        const devices = {
            power: [
                { id: 'wind-g1', name: '风机组 #1-5 (德阳基地)' },
                { id: 'wind-g2', name: '风机组 #6-10 (德阳基地)' },
                { id: 'wind-g3', name: '风机组 #11-15 (绵阳基地)' },
                { id: 'wind-g4', name: '风机组 #16-20 (绵阳基地)' },
                { id: 'wind-all', name: '所有风机组' },
                { id: 'solar-a1', name: '光伏阵列 A1区 (成都高新)' },
                { id: 'solar-a2', name: '光伏阵列 A2区 (成都高新)' },
                { id: 'solar-b1', name: '光伏阵列 B1区 (天府新区)' },
                { id: 'solar-b2', name: '光伏阵列 B2区 (天府新区)' },
                { id: 'solar-c1', name: '光伏阵列 C1区 (简阳基地)' },
                { id: 'solar-all', name: '所有光伏阵列' }
            ],
            storage: [
                { id: 'storage-1', name: '储能站 #1 (磷酸铁锂, 100MWh)' },
                { id: 'storage-2', name: '储能站 #2 (磷酸铁锂, 100MWh)' },
                { id: 'storage-3', name: '储能站 #3 (钒电池, 50MWh)' },
                { id: 'storage-4', name: '应急储能单元 (锂电池, 20MWh)' },
                { id: 'storage-g1', name: '高新区储能组 (160MWh)' },
                { id: 'storage-g2', name: '天府新区储能组 (110MWh)' },
                { id: 'storage-all', name: '所有储能单元' }
            ],
            load: [
                // 工业负荷
                { id: 'load-ind-1', name: '高新区工业园A (重工业)' },
                { id: 'load-ind-2', name: '高新区工业园B (轻工业)' },
                { id: 'load-ind-3', name: '天府新区工业园 (混合工业)' },
                { id: 'load-ind-all', name: '所有工业负荷' },
                // 商业负荷
                { id: 'load-com-1', name: '高新区商圈 (写字楼群)' },
                { id: 'load-com-2', name: '天府新区商圈 (商场群)' },
                { id: 'load-com-3', name: '成都东站商圈' },
                { id: 'load-com-all', name: '所有商业负荷' },
                // 民用负荷
                { id: 'load-res-1', name: '高新区住宅区A' },
                { id: 'load-res-2', name: '高新区住宅区B' },
                { id: 'load-res-3', name: '天府新区住宅区' },
                { id: 'load-res-all', name: '所有民用负荷' },
                // 特殊负荷
                { id: 'load-spec-1', name: '医院与应急设施' },
                { id: 'load-spec-2', name: '数据中心' },
                { id: 'load-spec-3', name: '轨道交通' },
                { id: 'load-all', name: '所有负荷' }
            ]
        };

        // 根据调度类型更新设备列表
        const updateDevices = (type) => {
            deviceSelect.innerHTML = '<option value="">选择设备...</option>';
            
            // 根据调度类型选择设备列表
            let deviceList = [];
            switch(type) {
                case 'power':
                    deviceList = devices.power;
                    break;
                case 'storage':
                    deviceList = devices.storage;
                    break;
                case 'load':
                    deviceList = devices.load;
                    break;
            }

            // 添加设备分组
            if (type === 'load') {
                // 工业负荷组
                let industrialGroup = document.createElement('optgroup');
                industrialGroup.label = '工业负荷';
                deviceList.filter(d => d.id.startsWith('load-ind-')).forEach(device => {
                    const option = document.createElement('option');
                    option.value = device.id;
                    option.textContent = device.name;
                    industrialGroup.appendChild(option);
                });
                deviceSelect.appendChild(industrialGroup);

                // 商业负荷组
                let commercialGroup = document.createElement('optgroup');
                commercialGroup.label = '商业负荷';
                deviceList.filter(d => d.id.startsWith('load-com-')).forEach(device => {
                    const option = document.createElement('option');
                    option.value = device.id;
                    option.textContent = device.name;
                    commercialGroup.appendChild(option);
                });
                deviceSelect.appendChild(commercialGroup);

                // 民用负荷组
                let residentialGroup = document.createElement('optgroup');
                residentialGroup.label = '民用负荷';
                deviceList.filter(d => d.id.startsWith('load-res-')).forEach(device => {
                    const option = document.createElement('option');
                    option.value = device.id;
                    option.textContent = device.name;
                    residentialGroup.appendChild(option);
                });
                deviceSelect.appendChild(residentialGroup);

                // 特殊负荷组
                let specialGroup = document.createElement('optgroup');
                specialGroup.label = '特殊负荷';
                deviceList.filter(d => d.id.startsWith('load-spec-') || d.id === 'load-all').forEach(device => {
                    const option = document.createElement('option');
                    option.value = device.id;
                    option.textContent = device.name;
                    specialGroup.appendChild(option);
                });
                deviceSelect.appendChild(specialGroup);
            } else if (type === 'power') {
                // 风电设备组
                let windGroup = document.createElement('optgroup');
                windGroup.label = '风力发电';
                deviceList.filter(d => d.id.startsWith('wind-')).forEach(device => {
                    const option = document.createElement('option');
                    option.value = device.id;
                    option.textContent = device.name;
                    windGroup.appendChild(option);
                });
                deviceSelect.appendChild(windGroup);

                // 光伏设备组
                let solarGroup = document.createElement('optgroup');
                solarGroup.label = '光伏发电';
                deviceList.filter(d => d.id.startsWith('solar-')).forEach(device => {
                    const option = document.createElement('option');
                    option.value = device.id;
                    option.textContent = device.name;
                    solarGroup.appendChild(option);
                });
                deviceSelect.appendChild(solarGroup);
            } else {
                // 储能设备组
                let storageGroup = document.createElement('optgroup');
                storageGroup.label = '储能设备';
                deviceList.forEach(device => {
                    const option = document.createElement('option');
                    option.value = device.id;
                    option.textContent = device.name;
                    storageGroup.appendChild(option);
                });
                deviceSelect.appendChild(storageGroup);
            }
        };

        // 初始化时更新一次
        const dispatchType = document.querySelector('#dispatchType');
        if (dispatchType) {
            updateDevices(dispatchType.value);
            // 监听类型变化
            dispatchType.addEventListener('change', (e) => updateDevices(e.target.value));
        }
    }

    // 处理调度类型变化
    handleDispatchTypeChange(type) {
        // 隐藏所有配置区域
        document.querySelector('#optimizationConfig')?.style.setProperty('display', 'none');
        document.querySelector('#storageConfig')?.style.setProperty('display', 'none');
        document.querySelector('#loadConfig')?.style.setProperty('display', 'none');

        // 显示对应的配置区域
        switch (type) {
            case 'power':
                document.querySelector('#optimizationConfig')?.style.setProperty('display', 'block');
                break;
            case 'storage':
                document.querySelector('#storageConfig')?.style.setProperty('display', 'block');
                break;
            case 'load':
                document.querySelector('#loadConfig')?.style.setProperty('display', 'block');
                break;
        }

        // 更新设备选择器
        this.initializeDeviceSelector();
    }

    // 处理调度确认
    handleDispatchConfirm() {
        const form = document.querySelector('#dispatchForm');
        if (!form) return;

        // 收集表单数据
        const formData = {
            type: form.querySelector('#dispatchType').value,
            device: form.querySelector('#targetDevice').value,
            executeTime: form.querySelector('#executeTime').value,
            priority: form.querySelector('#priority').value,
            note: form.querySelector('#dispatchNote').value
        };

        // 根据类型收集特定配置
        switch (formData.type) {
            case 'power':
                formData.optimization = {
                    target: form.querySelector('#optimizationTarget').value,
                    period: form.querySelector('#optimizationPeriod').value,
                    minPower: form.querySelector('#minPowerLimit').value,
                    maxPower: form.querySelector('#maxPowerLimit').value,
                    responseTime: form.querySelector('#responseTime').value,
                    adjustStep: form.querySelector('#adjustStep').value
                };
                break;
            case 'storage':
                formData.storage = {
                    mode: form.querySelector('#chargeMode').value,
                    power: form.querySelector('#storagePower').value,
                    minSoc: form.querySelector('#minSoc').value,
                    maxSoc: form.querySelector('#maxSoc').value
                };
                break;
            case 'load':
                formData.load = {
                    type: form.querySelector('#loadType').value,
                    mode: form.querySelector('#loadControlMode').value,
                    target: form.querySelector('#targetLoad').value,
                    responseTime: form.querySelector('#loadResponseTime').value
                };
                break;
        }

        // 验证必填字段
        if (!this.validateDispatchForm(formData)) {
            return;
        }

        // 创建调度任务
        this.createDispatchTask(formData);

        // 关闭模态框
        const modal = document.getElementById('dispatchModal');
        if (modal) {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
                
                // 清空表单
                form.reset();
            }
        }
    }

    // 验证调度表单
    validateDispatchForm(formData) {
        if (!formData.device) {
            this.showCustomNotification('warning', '请选择目标设备');
            return false;
        }

        if (!formData.executeTime) {
            this.showCustomNotification('warning', '请设置执行时间');
            return false;
        }

        // 根据类型验证特定字段
        switch (formData.type) {
            case 'power':
                if (!formData.optimization.minPower || !formData.optimization.maxPower) {
                    this.showCustomNotification('warning', '请设置功率限制范围');
                    return false;
                }
                break;
            case 'storage':
                if (!formData.storage.power) {
                    this.showCustomNotification('warning', '请设置储能功率');
                    return false;
                }
                break;
            case 'load':
                if (!formData.load.target) {
                    this.showCustomNotification('warning', '请设置目标负荷');
                    return false;
                }
                break;
        }

        return true;
    }

    // 创建调度任务
    createDispatchTask(formData) {
        const taskId = 'TASK-' + new Date().getTime().toString().slice(-6);
        const taskDescription = this.getTaskDescription(formData);
        this.addTaskToList(taskId, formData, taskDescription);
        this.showCustomNotification('success', '调度任务创建成功');

        const executeTime = new Date(formData.executeTime);
        if (executeTime <= new Date()) {
            this.executeDispatchTask(taskId, formData);
        }
    }

    // 获取任务描述
    getTaskDescription(formData) {
        const typeNames = {
            power: '功率调节',
            storage: '储能控制',
            load: '负荷控制'
        };

        let description = `${typeNames[formData.type]} - `;

        switch (formData.type) {
            case 'power':
                description += `优化目标: ${formData.optimization.target === 'efficiency' ? '效率优化' : 
                    formData.optimization.target === 'cost' ? '成本优化' : '负载均衡'}`;
                break;
            case 'storage':
                description += `${formData.storage.mode === 'charge' ? '充电' : '放电'} ${formData.storage.power}MW`;
                break;
            case 'load':
                description += `目标负荷: ${formData.load.target}MW`;
                break;
        }

        return description;
    }

    // 添加任务到列表
    addTaskToList(taskId, formData, description) {
        const tasksList = document.querySelector('#dispatchTasksList');
        if (!tasksList) return;

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${taskId}</td>
            <td>${description}</td>
            <td>${formData.device}</td>
            <td>${new Date(formData.executeTime).toLocaleString()}</td>
            <td><span class="badge bg-warning">待执行</span></td>
            <td>
                <div class="btn-group">
                    <button class="btn btn-sm btn-primary" onclick="window.powerDispatchControl.executeTask('${taskId}')">
                        执行
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="window.powerDispatchControl.cancelTask('${taskId}')">
                        取消
                    </button>
                </div>
            </td>
        `;

        tasksList.insertBefore(tr, tasksList.firstChild);
    }

    // 执行调度任务
    executeTask(taskId) {
        // 在实际应用中，这里会调用后端API执行任务
        // 这里只做界面更新演示
        const tr = document.querySelector(`tr:has(td:first-child:contains('${taskId}'))`);
        if (tr) {
            tr.querySelector('.badge').className = 'badge bg-success';
            tr.querySelector('.badge').textContent = '已完成';
            tr.querySelector('.btn-group').innerHTML = `
                <button class="btn btn-sm btn-secondary" onclick="window.powerDispatchControl.viewTaskDetail('${taskId}')">
                    详情
                </button>
            `;
            this.showCustomNotification('success', `任务 ${taskId} 执行成功`);
        }
    }

    // 取消调度任务
    cancelTask(taskId) {
        // 创建自定义确认对话框
        const confirmDialog = document.createElement('div');
        confirmDialog.className = 'custom-dialog';
        confirmDialog.innerHTML = `
            <div class="custom-dialog-content">
                <div class="dialog-header">
                    <h5>确认操作</h5>
                </div>
                <div class="dialog-body">
                    <p>确认要取消任务 ${taskId} 吗？</p>
                </div>
                <div class="dialog-footer">
                    <button class="btn btn-secondary" data-action="cancel">取消</button>
                    <button class="btn btn-danger" data-action="confirm">确认</button>
                </div>
            </div>
        `;

        document.body.appendChild(confirmDialog);
        setTimeout(() => confirmDialog.classList.add('show'), 10);

        confirmDialog.addEventListener('click', (e) => {
            const action = e.target.getAttribute('data-action');
            if (action === 'confirm') {
                const tr = document.querySelector(`tr:has(td:first-child:contains('${taskId}'))`);
                if (tr) {
                    tr.querySelector('.badge').className = 'badge bg-secondary';
                    tr.querySelector('.badge').textContent = '已取消';
                    tr.querySelector('.btn-group').innerHTML = `
                        <button class="btn btn-sm btn-secondary" onclick="window.powerDispatchControl.viewTaskDetail('${taskId}')">
                            详情
                        </button>
                    `;
                    this.showCustomNotification('info', `任务 ${taskId} 已取消`);
                }
            }
            confirmDialog.classList.remove('show');
            setTimeout(() => confirmDialog.remove(), 300);
        });
    }

    // 查看任务详情
    viewTaskDetail(taskId) {
        this.showCustomNotification('info', `查看任务 ${taskId} 的详细信息`);
    }
}

// 当DOM加载完成后初始化控制器
document.addEventListener('DOMContentLoaded', () => {
    window.powerDispatchControl = new PowerDispatchControl();
}); 