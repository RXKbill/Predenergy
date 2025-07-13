/**
 * 检修工单管理类
 */
class MaintenanceWorkOrder {
    constructor() {
        // 添加初始化检查
        if (!this.checkRequiredElements()) {
            console.warn('Required elements not found for MaintenanceWorkOrder');
            return;
        }

        // 初始化状态
        this.state = {
            currentWorkOrder: null,
            workOrders: new Map(),
            equipmentList: new Map(),
            filters: {
                type: 'all',
                status: 'all',
                priority: 'all',
                dateRange: null
            }
        };

        // 绑定方法到实例
        this.createWorkOrder = this.createWorkOrder.bind(this);
        this.addMaintenanceItem = this.addMaintenanceItem.bind(this);
        this.updateStandardValue = this.updateStandardValue.bind(this);
        this.removeMaintenanceItem = this.removeMaintenanceItem.bind(this);

        // 初始化配置
        this.config = {
            statusColors: {
                pending: 'info',
                in_progress: 'warning',
                completed: 'success',
                cancelled: 'danger',
                emergency: 'danger'
            },
            priorityLevels: {
                high: { label: '高', color: 'danger' },
                medium: { label: '中', color: 'warning' },
                low: { label: '低', color: 'info' }
            },
            maintenanceTypes: {
                routine: '例行检修',
                preventive: '预防性检修',
                corrective: '故障检修',
                predictive: '预测性检修'
            },
            equipmentTypes: {
                photovoltaicPanel: {
                    name: '光伏组件',
                    subTypes: ['单晶硅', '多晶硅', '薄膜'],
                    standardItems: [
                        {
                            name: '外观检查',
                            checkPoints: [
                                '组件表面清洁度',
                                '电池片破损、裂纹情况',
                                '焊带连接状态',
                                '边框密封完整性',
                                '背板鼓包、变色情况',
                                '接线盒密封性'
                            ]
                        },
                        {
                            name: '电气性能检测',
                            checkPoints: [
                                '开路电压测量',
                                '短路电流测量',
                                '绝缘电阻测试',
                                '旁路二极管功能测试',
                                '接地连接测试'
                            ]
                        },
                        {
                            name: '支架系统检查',
                            checkPoints: [
                                '支架结构完整性',
                                '螺栓紧固状态',
                                '防腐层状况',
                                '接地连接可靠性'
                            ]
                        }
                    ]
                },
                inverter: {
                    name: '逆变器',
                    subTypes: ['集中式', '组串式'],
                    standardItems: [
                        {
                            name: '外观及环境检查',
                            checkPoints: [
                                '外壳完整性',
                                '通风系统状态',
                                '防尘防水性能',
                                '环境温湿度',
                                '散热系统状态'
                            ]
                        },
                        {
                            name: '电气性能检测',
                            checkPoints: [
                                'DC输入电压检测',
                                'AC输出电压检测',
                                '效率测试',
                                'MPPT功能测试',
                                '保护功能测试',
                                '通信功能测试'
                            ]
                        }
                    ]
                },
                combinerBox: {
                    name: '汇流箱',
                    subTypes: ['直流汇流箱', '智能汇流箱'],
                    standardItems: [
                        {
                            name: '外观检查',
                            checkPoints: [
                                '箱体完整性',
                                '密封性能',
                                '防水性能',
                                '标识完整性'
                            ]
                        },
                        {
                            name: '电气检测',
                            checkPoints: [
                                '熔断器状态',
                                '防雷器状态',
                                '接线端子紧固度',
                                '绝缘电阻测试',
                                '接地电阻测试'
                            ]
                        }
                    ]
                },
                transformer: {
                    name: '变压器',
                    subTypes: ['油浸式', '干式'],
                    standardItems: [
                        {
                            name: '外观检查',
                            checkPoints: [
                                '外壳完整性',
                                '油位检查(油浸式)',
                                '呼吸器状态',
                                '接线端子状态',
                                '冷却系统状态'
                            ]
                        },
                        {
                            name: '电气测试',
                            checkPoints: [
                                '绝缘电阻测试',
                                '变比测试',
                                '空载电流测试',
                                '温升测试',
                                '局部放电测试'
                            ]
                        }
                    ]
                },
                monitoringSystem: {
                    name: '监控系统',
                    subTypes: ['环境监测', '设备监测', '安防监测'],
                    standardItems: [
                        {
                            name: '硬件检查',
                            checkPoints: [
                                '传感器状态',
                                '通信设备状态',
                                '供电系统状态',
                                '显示设备状态'
                            ]
                        },
                        {
                            name: '软件检查',
                            checkPoints: [
                                '数据采集功能',
                                '报警功能',
                                '数据存储功能',
                                '远程访问功能'
                            ]
                        }
                    ]
                },
                windTurbine: {
                    name: '风力发电机组',
                    subTypes: ['双馈异步', '永磁直驱', '半直驱'],
                    standardItems: [
                        {
                            name: '叶片系统检查',
                            checkPoints: [
                                '叶片外观完整性',
                                '叶片表面状态',
                                '叶尖制动器状态',
                                '螺栓紧固状态',
                                '防雷系统完整性',
                                '叶片角度传感器'
                            ]
                        },
                        {
                            name: '机舱系统检查',
                            checkPoints: [
                                '主轴承温度',
                                '齿轮箱油位和温度',
                                '制动系统状态',
                                '偏航系统状态',
                                '液压系统压力',
                                '冷却系统状态'
                            ]
                        },
                        {
                            name: '发电机系统检查',
                            checkPoints: [
                                '定子绕组温度',
                                '转子绕组状态',
                                '碳刷磨损情况',
                                '轴承润滑状态',
                                '冷却风扇运行',
                                '接线端子紧固度'
                            ]
                        },
                        {
                            name: '变速箱检查',
                            checkPoints: [
                                '油位和油质',
                                '轴承温度',
                                '齿轮啮合状态',
                                '润滑系统状态',
                                '振动水平',
                                '异响检测'
                            ]
                        },
                        {
                            name: '控制系统检查',
                            checkPoints: [
                                'PLC运行状态',
                                '传感器数据准确性',
                                '通信系统稳定性',
                                '报警系统功能',
                                '远程控制功能',
                                '数据记录功能'
                            ]
                        },
                        {
                            name: '塔筒结构检查',
                            checkPoints: [
                                '基础螺栓紧固度',
                                '塔筒垂直度',
                                '焊缝完整性',
                                '防腐层状态',
                                '爬梯安全装置',
                                '门锁系统状态'
                            ]
                        }
                    ]
                },
                windControl: {
                    name: '风机控制系统',
                    subTypes: ['主控系统', '变桨系统', '偏航系统'],
                    standardItems: [
                        {
                            name: '主控制器检查',
                            checkPoints: [
                                '系统启动测试',
                                '参数设置检查',
                                '通信接口测试',
                                '报警功能测试',
                                '数据采集功能',
                                '备份电源测试'
                            ]
                        },
                        {
                            name: '变桨系统检查',
                            checkPoints: [
                                '变桨电机状态',
                                '变桨轴承检查',
                                '编码器精度',
                                '变桨控制精度',
                                '紧急停机功能',
                                '备用电源状态'
                            ]
                        },
                        {
                            name: '偏航系统检查',
                            checkPoints: [
                                '偏航驱动电机',
                                '偏航制动器',
                                '偏航轴承状态',
                                '偏航角度检测',
                                '电缆扭转保护',
                                '润滑系统状态'
                            ]
                        }
                    ]
                },
                windSensor: {
                    name: '风机传感系统',
                    subTypes: ['风速传感器', '振动传感器', '温度传感器'],
                    standardItems: [
                        {
                            name: '风速风向仪检查',
                            checkPoints: [
                                '风速计校准',
                                '风向标校准',
                                '加热功能测试',
                                '信号输出检查',
                                '安装固定状态',
                                '防雷保护状态'
                            ]
                        },
                        {
                            name: '振动监测系统',
                            checkPoints: [
                                '传感器灵敏度',
                                '数据采集精度',
                                '报警阈值设置',
                                '信号传输质量',
                                '分析软件功能',
                                '存储空间检查'
                            ]
                        },
                        {
                            name: '温度监测系统',
                            checkPoints: [
                                '温度传感器校准',
                                '测量范围检查',
                                '响应时间测试',
                                '报警功能测试',
                                '数据记录功能',
                                '接线可靠性'
                            ]
                        }
                    ]
                }
            },
            workConditions: {
                weather: ['晴天', '多云', '阴天', '小雨', '大雨', '大风'],
                temperature: {
                    min: -20,
                    max: 45,
                    unit: '°C'
                },
                windSpeed: {
                    max: 12,
                    min: 0,
                    unit: 'm/s'
                },
                humidity: {
                    max: 95,
                    unit: '%'
                },
                height: {
                    max: 100,
                    unit: 'm'
                },
                visibility: {
                    min: 200,
                    unit: 'm'
                }
            },
            safetyRequirements: {
                personalProtection: [
                    '安全帽',
                    '绝缘手套',
                    '绝缘鞋',
                    '防护眼镜',
                    '安全带',
                    '防晒装备'
                ],
                tools: [
                    '绝缘工具',
                    '万用表',
                    '红外测温仪',
                    'IV曲线测试仪',
                    '绝缘电阻测试仪',
                    '接地电阻测试仪'
                ],
                procedures: [
                    '工作票审批',
                    '现场安全确认',
                    '设备停电验电',
                    '挂接地线',
                    '安全警示牌',
                    '工作区域围栏'
                ],
                heightWork: [
                    '高空作业证',
                    '安全绳',
                    '安全帽',
                    '防坠落装置',
                    '救援设备'
                ],
                specialTools: [
                    '扭矩扳手',
                    '振动分析仪',
                    '红外测温仪',
                    '超声波检测仪',
                    '油液分析仪'
                ],
                weatherLimits: [
                    '风速限制',
                    '能见度要求',
                    '雷电预警',
                    '结冰条件',
                    '极端温度'
                ]
            }
        };

        // 绑定事件处理器
        this.bindEventHandlers();
        
        // 初始化数据
        this.loadInitialData();

        // 设置为全局实例
        window.maintenanceWorkOrder = this;
    }

    /**
     * 检查必要的 DOM 元素是否存在
     */
    checkRequiredElements() {
        const requiredElements = [
            'equipment-maintenance',  // 设备检修管理标签页
            'newWorkOrderBtn',       // 新建工单按钮
            'maintenanceItemsTable', // 检修项目表格
            'orderType',            // 工单类型筛选器
            'orderStatus',          // 工单状态筛选器
            'orderPriority'         // 工单优先级筛选器
        ];

        const missingElements = requiredElements.filter(id => !document.getElementById(id));
        if (missingElements.length > 0) {
            console.warn('Missing required elements:', missingElements);
            return false;
        }
        return true;
    }

    /**
     * 绑定事件处理器
     */
    bindEventHandlers() {
        // 新建工单按钮
        document.getElementById('newWorkOrderBtn')?.addEventListener('click', () => {
            this.showNewWorkOrderModal();
        });

        // 工单筛选器
        document.getElementById('orderType')?.addEventListener('change', (e) => {
            this.updateFilters('type', e.target.value);
        });
        document.getElementById('orderStatus')?.addEventListener('change', (e) => {
            this.updateFilters('status', e.target.value);
        });
        document.getElementById('orderPriority')?.addEventListener('change', (e) => {
            this.updateFilters('priority', e.target.value);
        });

        // 设备类型选择联动
        document.getElementById('equipmentType')?.addEventListener('change', (e) => {
            this.updateEquipmentList(e.target.value);
        });

        // 检修项目动态添加
        document.querySelector('.btn-add-item')?.addEventListener('click', () => {
            this.addMaintenanceItem();
        });

        // 工单表单提交
        document.getElementById('newWorkOrderForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.createWorkOrder();
        });
    }

    /**
     * 加载初始数据
     */
    loadInitialData() {
        // 模拟加载设备列表
        this.loadEquipmentList();
        // 模拟加载工单数据
        this.loadWorkOrders();
    }

    /**
     * 加载设备列表
     */
    loadEquipmentList() {
        // 模拟设备数据
        const equipmentData = [
            { id: 'TRANS-001', name: '主变压器#1', type: 'transformer', status: 'normal' },
            { id: 'TRANS-002', name: '主变压器#2', type: 'transformer', status: 'normal' },
            { id: 'INV-001', name: '逆变器#1', type: 'inverter', status: 'warning' },
            { id: 'BAT-001', name: '储能电池组#1', type: 'battery', status: 'normal' },
            { id: 'PANEL-A1', name: '光伏组件A1区', type: 'panel', status: 'normal' },
            { id: 'WIND-001', name: '风机#1', type: 'windTurbine', status: 'normal' }
        ];

        equipmentData.forEach(equipment => {
            this.state.equipmentList.set(equipment.id, equipment);
        });
    }

    /**
     * 更新设备列表
     */
    updateEquipmentList(type) {
        const equipmentSelect = document.getElementById('equipmentId');
        if (!equipmentSelect) return;

        // 清空现有选项
        equipmentSelect.innerHTML = '<option value="">选择设备...</option>';

        // 根据类型筛选设备
        Array.from(this.state.equipmentList.values())
            .filter(equipment => equipment.type === type)
            .forEach(equipment => {
                const option = document.createElement('option');
                option.value = equipment.id;
                option.textContent = equipment.name;
                equipmentSelect.appendChild(option);
            });
    }

    /**
     * 显示新建工单模态框
     */
    showNewWorkOrderModal() {
        const modal = new bootstrap.Modal(document.getElementById('newWorkOrderModal'));
        modal.show();
    }

    /**
     * 创建工单
     */
    createWorkOrder() {
        console.log('开始创建工单...');
        const form = document.getElementById('newWorkOrderForm');
        if (!form) {
            console.error('表单不存在');
            this.showToast('error', '表单不存在');
            return;
        }

        try {
            // 验证必填字段
            if (!this.validateForm(form)) {
                console.warn('表单验证失败');
                return;
            }

            // 收集检修项目
            const maintenanceItems = this.collectMaintenanceItems();
            if (maintenanceItems.length === 0) {
                console.warn('没有检修项目');
                this.showToast('error', '请至少添加一个检修项目');
                return;
            }

            // 收集基本信息
            const workOrder = {
                id: `WO-${Date.now()}`,
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString(),
                status: 'pending',
                // 基本信息
                type: document.getElementById('maintenanceType').value,
                priority: document.getElementById('priority').value,
                // 设备信息
                equipment: {
                    type: document.getElementById('equipmentType').value,
                    model: document.getElementById('equipmentModel').value,
                    manufacturer: document.getElementById('manufacturer').value,
                    installationDate: document.getElementById('installationDate').value,
                    area: document.getElementById('equipmentArea').value,
                    location: {
                        specific: document.getElementById('specificLocation').value,
                        latitude: document.getElementById('latitude').value,
                        longitude: document.getElementById('longitude').value,
                        mountPosition: document.getElementById('mountPosition').value
                    },
                    status: {
                        current: document.getElementById('equipmentStatus').value,
                        operationHours: document.getElementById('operationHours').value,
                        faultDescription: document.getElementById('faultDescription').value
                    }
                },
                // 检修计划
                schedule: {
                    plannedStartTime: document.getElementById('plannedStartTime').value,
                    plannedEndTime: document.getElementById('plannedEndTime').value,
                    estimatedDuration: document.getElementById('estimatedDuration').value,
                    timeRequirement: document.getElementById('timeRequirement').value,
                    weatherRequirements: {
                        sunny: document.getElementById('weather_sunny').checked,
                        cloudy: document.getElementById('weather_cloudy').checked,
                        overcast: document.getElementById('weather_overcast').checked
                    },
                    maintenanceConditions: {
                        powerOff: document.getElementById('condition_power').checked,
                        scaffold: document.getElementById('condition_scaffold').checked,
                        crane: document.getElementById('condition_crane').checked
                    }
                },
                // 检修内容
                maintenanceItems: maintenanceItems,
                // 人员安排
                personnel: {
                    assignee: document.getElementById('assignee').value,
                    assistants: Array.from(document.getElementById('assistants').selectedOptions).map(opt => opt.value)
                },
                // 安全要求
                safetyRequirements: {
                    ppe: document.getElementById('safety_ppe').checked,
                    powerSafety: document.getElementById('safety_power').checked,
                    tools: document.getElementById('safety_tools').checked
                },
                // 备注
                notes: document.getElementById('notes').value || ''
            };

            console.log('工单数据:', workOrder);

            // 确保 workOrders 已初始化
            if (!this.state.workOrders) {
                this.state.workOrders = new Map();
            }

            // 保存工单
            this.state.workOrders.set(workOrder.id, workOrder);

            // 更新工单列表显示
            this.updateWorkOrdersList();

            // 关闭模态框
            const modal = bootstrap.Modal.getInstance(document.getElementById('newWorkOrderModal'));
            if (modal) {
                modal.hide();
            }

            // 重置表单
            form.reset();

            // 显示成功提示
            this.showToast('success', `工单创建成功，工单号：${workOrder.id}`);

            // 触发工单创建事件
            const event = new CustomEvent('workorder-created', { detail: workOrder });
            document.dispatchEvent(event);

        } catch (error) {
            console.error('创建工单失败:', error);
            this.showToast('error', '创建工单失败，请检查输入数据');
        }
    }

    /**
     * 验证表单
     */
    validateForm(form) {
        let isValid = true;
        let firstInvalidField = null;

        // 基本信息验证
        const requiredFields = {
            'maintenanceType': '请选择工单类型',
            'priority': '请选择优先级',
            'equipmentType': '请选择设备类型',
            'equipmentModel': '请输入设备型号',
            'manufacturer': '请输入制造商',
            'installationDate': '请选择安装日期',
            'equipmentArea': '请选择所属区域',
            'specificLocation': '请输入具体位置',
            'equipmentStatus': '请选择设备状态',
            'operationHours': '请输入运行时长',
            'plannedStartTime': '请选择计划开始时间',
            'plannedEndTime': '请选择计划结束时间',
            'estimatedDuration': '请输入预计工期',
            'assignee': '请选择负责人'
        };

        // 检查必填字段
        for (const [fieldId, errorMessage] of Object.entries(requiredFields)) {
            const field = document.getElementById(fieldId);
            if (!field) {
                console.warn(`字段 ${fieldId} 不存在`);
                continue;
            }

            const value = field.value.trim();
            if (!value) {
                isValid = false;
                field.classList.add('is-invalid');
                if (!firstInvalidField) {
                    firstInvalidField = field;
                    this.showToast('error', errorMessage);
                }
            } else {
                field.classList.remove('is-invalid');
            }
        }

        // 验证时间
        const startTime = document.getElementById('plannedStartTime');
        const endTime = document.getElementById('plannedEndTime');
        if (startTime && endTime && startTime.value && endTime.value) {
            const start = new Date(startTime.value);
            const end = new Date(endTime.value);
            if (end <= start) {
                isValid = false;
                endTime.classList.add('is-invalid');
                if (!firstInvalidField) {
                    firstInvalidField = endTime;
                    this.showToast('error', '计划结束时间必须晚于开始时间');
                }
            } else {
                endTime.classList.remove('is-invalid');
            }
        }

        // 验证检修项目
        const maintenanceTable = document.getElementById('maintenanceItemsTable');
        if (maintenanceTable) {
            const items = maintenanceTable.querySelectorAll('tbody tr');
            if (items.length === 0) {
                isValid = false;
                this.showToast('error', '请至少添加一个检修项目');
            } else {
                // 验证每个检修项目的完整性
                items.forEach(item => {
                    const itemSelect = item.querySelector('.standard-item-select');
                    const customItem = item.querySelector('.custom-item');
                    const standardValue = item.querySelector('.standard-value');
                    const inspectionMethod = item.querySelector('.inspection-method');

                    if (itemSelect && standardValue && inspectionMethod) {
                        const itemValue = itemSelect.value === 'custom' ? customItem.value : itemSelect.value;
                        if (!itemValue || !standardValue.value || !inspectionMethod.value) {
                            isValid = false;
                            if (!firstInvalidField) {
                                this.showToast('error', '请完整填写检修项目信息');
                            }
                        }
                    }
                });
            }
        }

        // 验证维修条件
        const conditions = ['condition_power', 'condition_scaffold', 'condition_crane'];
        const hasCondition = conditions.some(id => {
            const checkbox = document.getElementById(id);
            return checkbox && checkbox.checked;
        });
        if (!hasCondition) {
            isValid = false;
            if (!firstInvalidField) {
                this.showToast('error', '请至少选择一项维修条件');
            }
        }

        // 验证安全要求
        const safetyChecks = ['safety_ppe', 'safety_power', 'safety_tools'];
        const missingSafety = safetyChecks.some(id => {
            const checkbox = document.getElementById(id);
            return checkbox && !checkbox.checked;
        });
        if (missingSafety) {
            isValid = false;
            if (!firstInvalidField) {
                this.showToast('error', '请确认所有安全要求');
            }
        }

        if (!isValid && firstInvalidField) {
            firstInvalidField.focus();
            firstInvalidField.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        return isValid;
    }

    /**
     * 收集检修项目数据
     */
    collectMaintenanceItems() {
        const items = [];
        const tbody = document.querySelector('#maintenanceItemsTable tbody');
        
        if (!tbody) {
            console.warn('未找到检修项目表格');
            return items;
        }

        const rows = tbody.querySelectorAll('tr');
        if (rows.length === 0) {
            console.warn('检修项目表格为空');
            return items;
        }
        
        rows.forEach((row, index) => {
            try {
                // 获取输入元素
                const nameInput = row.querySelector('.standard-item-select, .custom-item:not(.d-none)');
                const standardInput = row.querySelector('.standard-value');
                const methodSelect = row.querySelector('.inspection-method');

                // 检查元素是否存在
                if (!nameInput || !standardInput) {
                    console.warn(`第${index + 1}行缺少必要的输入字段`, { nameInput, standardInput });
                    return;
                }

                // 获取项目名称（考虑自定义输入的情况）
                let itemName = '';
                if (nameInput.tagName === 'SELECT') {
                    if (nameInput.value === 'custom') {
                        const customInput = row.querySelector('.custom-item:not(.d-none)');
                        if (customInput) {
                            itemName = customInput.value;
                        }
                    } else {
                        itemName = nameInput.value;
                    }
                } else {
                    itemName = nameInput.value;
                }

                // 如果项目名称为空，跳过该项
                if (!itemName.trim()) {
                    console.warn(`第${index + 1}行项目名称为空`);
                    return;
                }

                const item = {
                    id: `item-${index + 1}`,
                    name: itemName,
                    standardValue: standardInput.value,
                    method: methodSelect ? methodSelect.value : '',
                    status: 'pending',
                    measuredValue: null,
                    completedAt: null
                };
                items.push(item);
            } catch (error) {
                console.error(`处理第${index + 1}行时出错:`, error);
            }
        });

        console.log('收集到的检修项目:', items);
        return items;
    }

    /**
     * 添加检修项目行
     */
    addMaintenanceItem() {
        const tbody = document.querySelector('#maintenanceItemsTable tbody');
        if (!tbody) {
            console.error('未找到检修项目表格');
            this.showToast('error', '无法添加检修项目，表格不存在');
            return;
        }

        const equipmentType = document.getElementById('equipmentType')?.value;
        let standardItems = [];
        
        // 根据设备类型获取标准检查项目
        if (equipmentType) {
            standardItems = this.getStandardItems(equipmentType);
        }

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <select class="form-select form-select-sm standard-item-select" onchange="maintenanceWorkOrder.updateStandardValue(this)">
                    <option value="">选择检查项目...</option>
                    ${standardItems.map(item => `
                        <option value="${item.name}">${item.name}</option>
                    `).join('')}
                    <option value="custom">自定义项目</option>
                </select>
                <input type="text" class="form-control form-control-sm mt-1 custom-item d-none" placeholder="输入检查项目">
            </td>
            <td>
                <input type="text" class="form-control form-control-sm standard-value" placeholder="输入标准值">
            </td>
            <td>
                <select class="form-select form-select-sm inspection-method">
                    <option value="">选择方法...</option>
                    <option value="visual">目视检查</option>
                    <option value="instrument">仪器测量</option>
                    <option value="thermal">红外测温</option>
                    <option value="electrical">电气测试</option>
                </select>
            </td>
            <td>
                <button type="button" class="btn btn-sm btn-danger" onclick="maintenanceWorkOrder.removeMaintenanceItem(this)">
                    <i class="ri-delete-bin-line"></i>
                </button>
            </td>
        `;
        tbody.appendChild(row);

        // 绑定事件
        const standardItemSelect = row.querySelector('.standard-item-select');
        if (standardItemSelect) {
            standardItemSelect.addEventListener('change', (e) => this.updateStandardValue(e.target));
        }
    }

    /**
     * 获取标准检查项目
     */
    getStandardItems(equipmentType) {
        const standardItems = {
            photovoltaic_panel: [
                { name: '组件表面清洁度检查', standard: '无明显污渍、积灰' },
                { name: '组件外观完整性检查', standard: '无破损、裂纹' },
                { name: '接线盒密封性检查', standard: '密封完好，无松动' },
                { name: '支架固定情况检查', standard: '牢固，无锈蚀' },
                { name: '开路电压测试', standard: 'Voc ± 5%' },
                { name: '短路电流测试', standard: 'Isc ± 5%' },
                { name: '绝缘电阻测试', standard: '≥ 100MΩ' },
                { name: '热斑检测', standard: '温差 ≤ 20℃' }
            ],
            inverter: [
                { name: '外观检查', standard: '外壳完整，无变形' },
                { name: '散热系统检查', standard: '风扇运转正常' },
                { name: 'DC输入电压检测', standard: '电压在允许范围内' },
                { name: 'AC输出电压检测', standard: '电压波动≤±5%' },
                { name: '效率测试', standard: '≥98%' },
                { name: '通信功能测试', standard: '通信正常' },
                { name: '防护等级检查', standard: 'IP65达标' }
            ],
            combiner_box: [
                { name: '外壳密封性检查', standard: '密封完好' },
                { name: '接地连接检查', standard: '接地电阻≤4Ω' },
                { name: '熔断器检查', standard: '完好无损' },
                { name: '防雷器状态检查', standard: '指示正常' },
                { name: '端子紧固性检查', standard: '无松动' }
            ]
        };

        return standardItems[equipmentType] || [];
    }

    /**
     * 更新标准值
     */
    updateStandardValue(select) {
        if (!select) return;

        const row = select.closest('tr');
        if (!row) return;

        const standardValueInput = row.querySelector('.standard-value');
        const customInput = row.querySelector('.custom-item');

        if (!standardValueInput || !customInput) {
            console.warn('未找到必要的输入字段');
            return;
        }

        if (select.value === 'custom') {
            customInput.classList.remove('d-none');
            select.classList.add('d-none');
            standardValueInput.value = '';
        } else {
            const equipmentType = document.getElementById('equipmentType')?.value;
            if (equipmentType) {
                const standardItems = this.getStandardItems(equipmentType);
                const selectedItem = standardItems.find(item => item.name === select.value);
                if (selectedItem) {
                    standardValueInput.value = selectedItem.standard;
                }
            }
        }
    }

    /**
     * 删除检修项目行
     */
    removeMaintenanceItem(button) {
        button.closest('tr').remove();
    }

    /**
     * 更新工单列表显示
     */
    updateWorkOrdersList() {
        const tbody = document.querySelector('.work-orders-table tbody');
        if (!tbody) return;

        // 清空现有内容
        tbody.innerHTML = '';

        // 应用筛选
        const filteredOrders = this.getFilteredWorkOrders();

        // 生成工单行
        filteredOrders.forEach(order => {
            const row = this.createWorkOrderRow(order);
            tbody.appendChild(row);
        });
    }

    /**
     * 获取筛选后的工单列表
     */
    getFilteredWorkOrders() {
        return Array.from(this.state.workOrders.values())
            .filter(order => {
                const { type, status, priority } = this.state.filters;
                return (type === 'all' || order.type === type) &&
                       (status === 'all' || order.status === status) &&
                       (priority === 'all' || order.priority === priority);
            });
    }

    /**
     * 创建工单行
     */
    createWorkOrderRow(order) {
        const equipment = this.state.equipmentList.get(order.equipmentId);
        const row = document.createElement('tr');
        
        row.innerHTML = `
            <td>${order.id}</td>
            <td>${this.config.maintenanceTypes[order.type]}</td>
            <td>${equipment ? equipment.name : '未知设备'}</td>
            <td>${order.plannedStartTime}</td>
            <td>${order.assignee}</td>
            <td><span class="badge bg-${this.config.statusColors[order.status]}">${this.getStatusText(order.status)}</span></td>
            <td>
                <div class="btn-group">
                    <button class="btn btn-sm btn-info" onclick="maintenanceWorkOrder.viewWorkOrder('${order.id}')">
                        <i class="ri-eye-line"></i>
                    </button>
                    <button class="btn btn-sm btn-success" onclick="maintenanceWorkOrder.updateWorkOrder('${order.id}')">
                        <i class="ri-edit-line"></i>
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="maintenanceWorkOrder.cancelWorkOrder('${order.id}')">
                        <i class="ri-close-line"></i>
                    </button>
                </div>
            </td>
        `;

        return row;
    }

    /**
     * 查看工单详情
     */
    viewWorkOrder(orderId) {
        const order = this.state.workOrders.get(orderId);
        if (!order) return;

        // 设置当前工单
        this.state.currentWorkOrder = order;

        // 更新模态框内容
        this.updateWorkOrderDetailModal(order);

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('workOrderDetailModal'));
        modal.show();
    }

    /**
     * 更新工单详情模态框
     */
    updateWorkOrderDetailModal(order) {
        // 更新工单ID
        document.getElementById('workOrderId').textContent = order.id;

        // 更新基本信息
        document.getElementById('detailMaintenanceType').textContent = this.config.maintenanceTypes[order.type];
        document.getElementById('detailPriority').textContent = this.config.priorityLevels[order.priority].label;
        document.getElementById('detailPlannedTime').textContent = order.plannedStartTime;
        document.getElementById('detailDuration').textContent = `${order.estimatedDuration}小时`;

        // 更新设备信息
        const equipment = this.state.equipmentList.get(order.equipmentId);
        if (equipment) {
            document.getElementById('detailEquipmentName').textContent = equipment.name;
            document.getElementById('detailEquipmentId').textContent = equipment.id;
        }

        // 更新检修项目列表
        this.updateMaintenanceItemsList(order.items);

        // 更新工作记录
        this.updateWorkLogs(order.id);
    }

    /**
     * 更新检修项目列表
     */
    updateMaintenanceItemsList(items) {
        const tbody = document.getElementById('detailItemsList');
        if (!tbody) return;

        tbody.innerHTML = items.map(item => `
            <tr>
                <td>${item.name}</td>
                <td>${item.standardValue}</td>
                <td>${item.measuredValue || '-'}</td>
                <td><span class="badge bg-${this.getStatusColor(item.status)}">${this.getStatusText(item.status)}</span></td>
                <td>${item.completedAt || '-'}</td>
                <td>
                    <button class="btn btn-sm btn-${item.status === 'pending' ? 'primary' : 'light'}" 
                            onclick="maintenanceWorkOrder.viewItemDetail('${item.id}')">
                        <i class="ri-${item.status === 'pending' ? 'edit' : 'eye'}-line"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    }

    /**
     * 更新工作记录
     */
    updateWorkLogs(orderId) {
        const container = document.getElementById('detailWorkLogs');
        if (!container) return;

        // 获取工单的工作记录
        const logs = this.getWorkLogs(orderId);

        container.innerHTML = logs.map(log => `
            <div class="log-item">
                <div class="log-time">${this.formatTime(log.timestamp)}</div>
                <div class="log-content">
                    <h6>${log.title}</h6>
                    <p>${log.content}</p>
                    <div class="log-meta">
                        <span class="log-user">${log.user}</span>
                        <span class="log-location">${log.location}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }

    /**
     * 获取工作记录
     */
    getWorkLogs(orderId) {
        // 模拟获取工作记录
        return [
            {
                id: 'log1',
                timestamp: new Date('2024-04-01T09:00:00'),
                title: '开始检修工作',
                content: '按计划开始主变压器#1的例行检修工作',
                user: '张工',
                location: '变电站A区'
            },
            {
                id: 'log2',
                timestamp: new Date('2024-04-01T09:15:00'),
                title: '完成绝缘电阻测量',
                content: '测量值: 1200MΩ，状态正常',
                user: '张工',
                location: '变电站A区'
            },
            {
                id: 'log3',
                timestamp: new Date('2024-04-01T09:30:00'),
                title: '完成油温检测',
                content: '测量值: 58℃，状态正常',
                user: '李工',
                location: '变电站A区'
            }
        ];
    }

    /**
     * 查看检查项目详情
     */
    viewItemDetail(itemId) {
        const order = this.state.currentWorkOrder;
        if (!order) return;

        const item = order.items.find(i => i.id === itemId);
        if (!item) return;

        // 更新模态框内容
        document.getElementById('itemName').value = item.name;
        document.getElementById('standardValue').value = item.standardValue;
        document.getElementById('measuredValue').value = item.measuredValue || '';
        document.getElementById('itemStatus').value = item.status;
        document.getElementById('itemNotes').value = item.notes || '';

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('itemDetailModal'));
        modal.show();
    }

    /**
     * 保存检查项目详情
     */
    saveItemDetail() {
        const order = this.state.currentWorkOrder;
        if (!order) return;

        const itemName = document.getElementById('itemName').value;
        const item = order.items.find(i => i.name === itemName);
        if (!item) return;

        // 更新项目数据
        item.measuredValue = document.getElementById('measuredValue').value;
        item.status = document.getElementById('itemStatus').value;
        item.notes = document.getElementById('itemNotes').value;
        item.completedAt = item.status === 'completed' ? new Date().toISOString() : null;

        // 更新工单状态
        this.updateWorkOrderStatus(order);

        // 更新显示
        this.updateMaintenanceItemsList(order.items);

        // 关闭模态框
        bootstrap.Modal.getInstance(document.getElementById('itemDetailModal')).hide();

        // 显示成功提示
        this.showToast('success', '检查项目更新成功');
    }

    /**
     * 更新工单状态
     */
    updateWorkOrderStatus(order) {
        const totalItems = order.items.length;
        const completedItems = order.items.filter(item => item.status === 'completed').length;
        const pendingItems = order.items.filter(item => item.status === 'pending').length;

        if (completedItems === totalItems) {
            order.status = 'completed';
        } else if (pendingItems === totalItems) {
            order.status = 'pending';
        } else {
            order.status = 'in_progress';
        }

        // 更新工单列表显示
        this.updateWorkOrdersList();
    }

    /**
     * 添加工作记录
     */
    addWorkLog() {
        // 显示添加工作记录模态框
        const modal = new bootstrap.Modal(document.getElementById('addWorkLogModal'));
        modal.show();
    }

    /**
     * 保存工作记录
     */
    saveWorkLog() {
        const order = this.state.currentWorkOrder;
        if (!order) return;

        const logType = document.getElementById('logType').value;
        const logTitle = document.getElementById('logTitle').value;
        const logContent = document.getElementById('logContent').value;

        // 创建新的工作记录
        const log = {
            id: `log-${Date.now()}`,
            timestamp: new Date(),
            type: logType,
            title: logTitle,
            content: logContent,
            user: '当前用户',
            location: '当前位置'
        };

        // 添加到工作记录列表
        if (!order.logs) order.logs = [];
        order.logs.push(log);

        // 更新显示
        this.updateWorkLogs(order.id);

        // 关闭模态框
        bootstrap.Modal.getInstance(document.getElementById('addWorkLogModal')).hide();

        // 显示成功提示
        this.showToast('success', '工作记录添加成功');
    }

    /**
     * 显示提示消息
     */
    showToast(type, message) {
        // 创建toast元素
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');

        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;

        // 添加到容器
        const container = document.getElementById('toast-container') || document.body;
        container.appendChild(toast);

        // 显示toast
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();

        // 自动移除
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    /**
     * 获取状态文本
     */
    getStatusText(status) {
        const statusMap = {
            pending: '待处理',
            in_progress: '进行中',
            completed: '已完成',
            cancelled: '已取消',
            emergency: '紧急'
        };
        return statusMap[status] || status;
    }

    /**
     * 获取状态颜色
     */
    getStatusColor(status) {
        return this.config.statusColors[status] || 'secondary';
    }

    /**
     * 格式化时间
     */
    formatTime(date) {
        if (typeof date === 'string') {
            date = new Date(date);
        }
        return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    }

    /**
     * 获取风机特定的标准检查项目
     */
    getWindTurbineStandardItems(subType) {
        const items = [];
        const equipment = this.config.equipmentTypes.windTurbine;
        
        if (equipment && equipment.standardItems) {
            equipment.standardItems.forEach(category => {
                category.checkPoints.forEach(point => {
                    items.push({
                        name: `${category.name} - ${point}`,
                        standard: this.getWindTurbineStandard(category.name, point),
                        method: this.getInspectionMethod(category.name, point)
                    });
                });
            });
        }
        
        return items;
    }

    /**
     * 获取风机检查项目的标准值
     */
    getWindTurbineStandard(category, point) {
        const standards = {
            '叶片系统检查': {
                '叶片外观完整性': '无裂纹、无变形',
                '叶片表面状态': '无明显磨损、无脱层',
                '螺栓紧固状态': '扭矩符合要求，无松动',
                '防雷系统完整性': '接地电阻≤4Ω'
            },
            '机舱系统检查': {
                '主轴承温度': '≤85℃',
                '齿轮箱油位和温度': '油位正常，温度≤75℃',
                '液压系统压力': '符合设计值±5%'
            },
            '发电机系统检查': {
                '定子绕组温度': '≤155℃',
                '碳刷磨损情况': '磨损≤50%',
                '轴承润滑状态': '润滑充足，无异响'
            }
        };

        return standards[category]?.[point] || '符合设计规范';
    }

    /**
     * 获取检查方法
     */
    getInspectionMethod(category, point) {
        const methods = {
            '叶片系统检查': {
                '叶片外观完整性': 'visual',
                '叶片表面状态': 'visual,ultrasonic',
                '防雷系统完整性': 'electrical'
            },
            '机舱系统检查': {
                '主轴承温度': 'thermal',
                '齿轮箱油位和温度': 'instrument',
                '液压系统压力': 'instrument'
            },
            '发电机系统检查': {
                '定子绕组温度': 'thermal',
                '碳刷磨损情况': 'visual,instrument',
                '轴承润滑状态': 'acoustic,vibration'
            }
        };

        return methods[category]?.[point] || 'visual';
    }
}

// 修改初始化方式
window.MaintenanceWorkOrder = MaintenanceWorkOrder;

// 确保只创建一个实例
function initializeMaintenanceWorkOrder() {
    if (!window.maintenanceWorkOrder) {
        console.log('正在初始化 MaintenanceWorkOrder...');
        window.maintenanceWorkOrder = new MaintenanceWorkOrder();
        console.log('MaintenanceWorkOrder 初始化完成');
    }
}

// 当 DOM 加载完成后初始化实例
document.addEventListener('DOMContentLoaded', () => {
    // 等待模态框加载完成
    const checkModalLoaded = setInterval(() => {
        const modalElement = document.getElementById('newWorkOrderModal');
        const createButton = document.getElementById('createWorkOrderBtn');
        
        if (modalElement && createButton) {
            clearInterval(checkModalLoaded);
            console.log('模态框和按钮已加载');
            
            // 初始化实例
            initializeMaintenanceWorkOrder();
            
            // 确保事件绑定
            createButton.addEventListener('click', () => {
                console.log('创建工单按钮被点击');
                if (window.maintenanceWorkOrder) {
                    window.maintenanceWorkOrder.createWorkOrder();
                } else {
                    console.error('maintenanceWorkOrder 实例未找到');
                }
            });
        }
    }, 100);
});

// 添加调试日志
console.log('maintenance-work-order.js loaded'); 