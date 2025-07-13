/**
 * 设备检修管理模块
 */
class MaintenanceManagement {
    constructor() {
        // 检查单例模式
        if (window.maintenanceManagementInstance) {
            return window.maintenanceManagementInstance;
        }
        window.maintenanceManagementInstance = this;

        // 初始化配置
        this.config = {
            priorityLevels: {
                urgent: { label: '紧急', color: '#dc3545' },
                high: { label: '高优先级', color: '#fd7e14' },
                normal: { label: '普通', color: '#0dcaf0' },
                low: { label: '低优先级', color: '#198754' }
            },
            maintenanceTypes: {
                routine: '例行检修',
                preventive: '预防性检修',
                corrective: '故障检修',
                predictive: '预测性检修'
            },
            equipmentCategories: {
                transformer: '变压器',
                switchgear: '开关设备',
                generator: '发电机组',
                inverter: '逆变器',
                battery: '储能电池',
                cable: '电缆线路'
            }
        };

        // 初始化状态
        this.state = {
            currentView: 'dashboard',
            selectedDate: new Date(),
            selectedEquipment: null,
            filters: {
                type: 'all',
                status: 'all',
                priority: 'all',
                dateRange: {
                    start: null,
                    end: null
                }
            }
        };

        // 无人机配置
        this.droneConfig = {
            inspectionTypes: {
                routine: { label: '常规巡检', icon: 'ri-route-line' },
                thermal: { label: '红外测温', icon: 'ri-temp-hot-line' },
                visual: { label: '可见光成像', icon: 'ri-camera-lens-line' },
                emergency: { label: '应急巡检', icon: 'ri-alarm-warning-line' }
            },
            areas: {
                area1: { label: '光伏区域A', coordinates: [[116.123, 39.456], [116.234, 39.567]] },
                area2: { label: '变电站B', coordinates: [[116.345, 39.678], [116.456, 39.789]] },
                area3: { label: '输电线路C', coordinates: [[116.567, 39.890], [116.678, 39.901]] }
            }
        };

        // 无人机状态
        this.droneStatus = {
            deviceId: 'DRONE-001',
            battery: 85,
            status: 'standby', // standby, flying, charging, maintenance
            flightTime: 25,
            position: [116.123, 39.456],
            altitude: 0,
            speed: 0
        };

        // 无人机集群配置
        this.droneFleetConfig = {
            fleets: {
                fleet1: {
                    id: 'FLEET-001',
                    name: '光伏巡检集群A',
                    drones: [
                        { id: 'DRONE-001', model: 'DJI M300 RTK', status: 'standby', battery: 85, role: 'leader' },
                        { id: 'DRONE-002', model: 'DJI M300 RTK', status: 'standby', battery: 90, role: 'follower' },
                        { id: 'DRONE-003', model: 'DJI M300 RTK', status: 'charging', battery: 30, role: 'follower' }
                    ],
                    capabilities: ['thermal', 'visual', 'lidar']
                },
                fleet2: {
                    id: 'FLEET-002',
                    name: '输电线路巡检集群B',
                    drones: [
                        { id: 'DRONE-004', model: 'DJI M350 RTK', status: 'standby', battery: 95, role: 'leader' },
                        { id: 'DRONE-005', model: 'DJI M350 RTK', status: 'standby', battery: 88, role: 'follower' },
                        { id: 'DRONE-006', model: 'DJI M350 RTK', status: 'maintenance', battery: 0, role: 'follower' }
                    ],
                    capabilities: ['corona', 'infrared', 'visual']
                }
            },
            inspectionModes: {
                sequential: { label: '顺序巡检', icon: 'ri-route-line' },
                parallel: { label: '并行巡检', icon: 'ri-split-cells-horizontal' },
                collaborative: { label: '协同巡检', icon: 'ri-team-line' }
            },
            sensorTypes: {
                visual: { label: '可见光相机', resolution: '4K' },
                thermal: { label: '红外热像仪', resolution: '640x512' },
                lidar: { label: '激光雷达', range: '100m' },
                corona: { label: '紫外成像仪', sensitivity: '3pC' },
                infrared: { label: '红外成像仪', resolution: '1280x1024' }
            },
            inspectionPatterns: {
                zigzag: { label: 'Z字形', description: '适用于大面积区域巡检' },
                circular: { label: '环形', description: '适用于塔架巡检' },
                linear: { label: '线性', description: '适用于输电线路巡检' },
                spiral: { label: '螺旋形', description: '适用于风机塔筒巡检' }
            }
        };

        // 无人机集群状态
        this.fleetStatus = {
            activeFleet: 'fleet1',
            totalDrones: 6,
            availableDrones: 4,
            chargingDrones: 1,
            maintenanceDrones: 1,
            weatherConditions: {
                temperature: 25,
                windSpeed: 3.5,
                visibility: 'good',
                precipitation: 0
            },
            flightRestrictions: {
                maxHeight: 120,
                maxSpeed: 15,
                noFlyZones: [
                    { name: '变电站A', coordinates: [[116.123, 39.456], [116.234, 39.567]] },
                    { name: '居民区B', coordinates: [[116.345, 39.678], [116.456, 39.789]] }
                ]
            }
        };

        // 状态数据
        this.stats = {
            totalDevices: 128,
            normalDevices: 108,
            warningDevices: 15,
            faultDevices: 5,
            healthRate: 96.8
        };

        // 初始化组件
        this.initializeComponents();
        this.bindEvents();
        this.loadDashboardData();
        this.initializeDroneFeatures();

        this.workOrders = new Map();
        this.drones = new Map();
        this.equipmentStatus = new Map();
        this.initialize();
    }

    async initialize() {
        try {
            await this.loadInitialData();
            this.initializeEventListeners();
            this.initializeCharts();
            this.startRealTimeUpdates();
            console.log('设备检修管理系统初始化完成');
        } catch (error) {
            console.error('设备检修管理系统初始化失败:', error);
        }
    }

    async loadInitialData() {
        // 模拟加载初始数据
        this.loadEquipmentStatus();
        this.loadWorkOrders();
        this.loadDroneStatus();
        this.updateDashboard();
    }

    loadEquipmentStatus() {
        // 模拟设备状态数据
        const equipmentData = [
            { id: 'EQ001', name: '主变压器#1', type: 'transformer', status: 'normal', lastMaintenance: '2024-01-15', runningDays: 3650 },
            { id: 'EQ002', name: '配电柜#A3', type: 'switchgear', status: 'warning', lastMaintenance: '2024-02-20', runningDays: 1825 },
            { id: 'EQ003', name: '储能系统#2', type: 'battery', status: 'normal', lastMaintenance: '2024-03-01', runningDays: 365 },
            // ... 更多设备数据
        ];

        equipmentData.forEach(equipment => {
            this.equipmentStatus.set(equipment.id, equipment);
        });
    }

    loadWorkOrders() {
        // 模拟工单数据
        const workOrders = [
            {
                id: 'WO001',
                title: '主变压器年度检修',
                type: 'routine',
                equipment: 'EQ001',
                status: 'in-progress',
                priority: 'high',
                startTime: '2024-03-26 09:00',
                endTime: '2024-03-26 17:00',
                assignedTo: ['张工', '李工'],
                tasks: [
                    { name: '外观检查', status: 'completed' },
                    { name: '绝缘测试', status: 'in-progress' },
                    { name: '油质检测', status: 'pending' }
                ]
            },
            // ... 更多工单数据
        ];

        workOrders.forEach(order => {
            this.workOrders.set(order.id, order);
        });
    }

    loadDroneStatus() {
        // 模拟无人机状态数据
        const drones = [
            {
                id: 'DRONE001',
                name: '巡检无人机#1',
                status: 'standby',
                battery: 85,
                flightTime: 25,
                lastMaintenance: '2024-03-20',
                currentTask: null
            },
            {
                id: 'DRONE002',
                name: '巡检无人机#2',
                status: 'in-mission',
                battery: 65,
                flightTime: 15,
                lastMaintenance: '2024-03-18',
                currentTask: {
                    id: 'TASK001',
                    area: '变电站A区',
                    progress: 75,
                    startTime: '2024-03-26 10:30'
                }
            }
        ];

        drones.forEach(drone => {
            this.drones.set(drone.id, drone);
        });
    }

    updateDashboard() {
        this.updateStatistics();
        this.updateEquipmentStatus();
        this.updateWorkOrdersList();
        this.updateDroneStatus();
    }

    updateStatistics() {
        // 更新统计数据
        const stats = {
            totalOrders: this.workOrders.size,
            pendingOrders: Array.from(this.workOrders.values()).filter(o => o.status === 'pending').length,
            completedOrders: Array.from(this.workOrders.values()).filter(o => o.status === 'completed').length,
            equipmentHealth: this.calculateEquipmentHealth()
        };

        // 更新DOM
        document.getElementById('total-orders').textContent = stats.totalOrders;
        document.getElementById('pending-orders').textContent = stats.pendingOrders;
        document.getElementById('completed-orders').textContent = stats.completedOrders;
        document.getElementById('equipment-health').textContent = stats.equipmentHealth + '%';
    }

    calculateEquipmentHealth() {
        const equipment = Array.from(this.equipmentStatus.values());
        const healthyCount = equipment.filter(e => e.status === 'normal').length;
        return Math.round((healthyCount / equipment.length) * 100);
    }

    updateEquipmentStatus() {
        const container = document.querySelector('.equipment-overview');
        if (!container) return;

        container.innerHTML = '';
        const statusCounts = {
            normal: 0,
            warning: 0,
            danger: 0
        };

        this.equipmentStatus.forEach(equipment => {
            statusCounts[equipment.status]++;
        });

        // 创建状态卡片
        Object.entries(statusCounts).forEach(([status, count]) => {
            const card = this.createStatusCard(status, count);
            container.appendChild(card);
        });
    }

    createStatusCard(status, count) {
        const card = document.createElement('div');
        card.className = 'equipment-status-card';
        
        const statusText = {
            normal: '正常运行',
            warning: '需要维护',
            danger: '故障设备'
        };

        card.innerHTML = `
            <div class="status-icon ${status}">
                <i class="ri-${status === 'normal' ? 'checkbox-circle' : status === 'warning' ? 'error-warning' : 'alarm-warning'}-line"></i>
            </div>
            <h4>${statusText[status]}</h4>
            <h2>${count}</h2>
            <div class="equipment-progress">
                <div class="progress" style="height: 6px;">
                    <div class="progress-bar bg-${status === 'normal' ? 'success' : status === 'warning' ? 'warning' : 'danger'}" 
                         style="width: ${(count / this.equipmentStatus.size * 100).toFixed(1)}%"></div>
                </div>
                <small class="text-muted">占比 ${(count / this.equipmentStatus.size * 100).toFixed(1)}%</small>
            </div>
        `;

        return card;
    }

    updateWorkOrdersList() {
        const container = document.querySelector('.work-orders-list');
        if (!container) return;

        container.innerHTML = '';
        this.workOrders.forEach(order => {
            const card = this.createWorkOrderCard(order);
            container.appendChild(card);
        });
    }

    createWorkOrderCard(order) {
        const card = document.createElement('div');
        card.className = 'work-order-card';
        
        const equipment = this.equipmentStatus.get(order.equipment);
        const completedTasks = order.tasks.filter(t => t.status === 'completed').length;
        const progress = (completedTasks / order.tasks.length * 100).toFixed(0);

        card.innerHTML = `
            <div class="work-order-header">
                <div>
                    <h6 class="work-order-title">${order.title}</h6>
                    <span class="work-order-id">#${order.id}</span>
                </div>
                <span class="work-order-status status-${order.status}">${this.getStatusText(order.status)}</span>
            </div>
            <div class="work-order-content">
                <div class="content-row">
                    <span class="content-label">设备:</span>
                    <span class="content-value">${equipment ? equipment.name : '未知设备'}</span>
                </div>
                <div class="content-row">
                    <span class="content-label">开始时间:</span>
                    <span class="content-value">${order.startTime}</span>
                </div>
                <div class="content-row">
                    <span class="content-label">进度:</span>
                    <div class="progress" style="height: 6px; width: 100px;">
                        <div class="progress-bar bg-primary" style="width: ${progress}%"></div>
                    </div>
                </div>
            </div>
            <div class="work-order-footer">
                <div class="assigned-users">
                    ${order.assignedTo.map(user => `
                        <div class="user-avatar">
                            <img src="static/picture/avatar-${Math.floor(Math.random() * 4) + 1}.jpg" alt="${user}">
                        </div>
                    `).join('')}
                </div>
                <div class="action-buttons">
                    <button class="btn btn-sm btn-primary" onclick="maintenanceManagement.viewWorkOrder('${order.id}')">
                        <i class="ri-eye-line me-1"></i>查看
                    </button>
                    ${order.status !== 'completed' ? `
                        <button class="btn btn-sm btn-success" onclick="maintenanceManagement.updateWorkOrder('${order.id}')">
                            <i class="ri-check-line me-1"></i>更新
                        </button>
                    ` : ''}
                </div>
            </div>
        `;

        return card;
    }

    updateDroneStatus() {
        const container = document.querySelector('.drone-status');
        if (!container) return;

        const activeDrone = Array.from(this.drones.values()).find(d => d.status === 'in-mission');
        
        container.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h6 class="mb-0">无人机状态</h6>
                <div class="drone-battery">
                    <i class="ri-battery-2-line me-1"></i>
                    <span>${activeDrone ? activeDrone.battery : '85'}%</span>
                </div>
            </div>
            <div class="drone-info">
                <div class="info-item">
                    <span class="info-label">设备编号:</span>
                    <span class="info-value">${activeDrone ? activeDrone.id : 'DRONE001'}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">当前状态:</span>
                    <span class="info-value ${activeDrone ? 'text-warning' : 'text-success'}">
                        ${activeDrone ? '任务中' : '待命中'}
                    </span>
                </div>
                <div class="info-item">
                    <span class="info-label">剩余续航:</span>
                    <span class="info-value">${activeDrone ? activeDrone.flightTime : '25'}分钟</span>
                </div>
                <div class="info-item">
                    <span class="info-label">上次维护:</span>
                    <span class="info-value">${activeDrone ? activeDrone.lastMaintenance : '2024-03-20'}</span>
                </div>
            </div>
        `;

        // 更新任务列表
        if (activeDrone && activeDrone.currentTask) {
            this.updateDroneTask(activeDrone.currentTask);
        }
    }

    updateDroneTask(task) {
        const container = document.querySelector('.inspection-tasks');
        if (!container) return;

        container.innerHTML = `
            <div class="task-card">
                <div class="task-header">
                    <h6 class="task-title">${task.area}巡检任务</h6>
                    <span class="task-status status-in-progress">进行中</span>
                </div>
                <div class="task-details">
                    开始时间: ${task.startTime}
                </div>
                <div class="task-progress">
                    <div class="progress-bar">
                        <div class="progress-value" style="width: ${task.progress}%"></div>
                    </div>
                    <small class="text-muted mt-1 d-block">完成度: ${task.progress}%</small>
                </div>
            </div>
        `;
    }

    getStatusText(status) {
        const statusMap = {
            'pending': '待处理',
            'in-progress': '进行中',
            'completed': '已完成',
            'emergency': '紧急'
        };
        return statusMap[status] || status;
    }

    viewWorkOrder(orderId) {
        const order = this.workOrders.get(orderId);
        if (!order) return;

        // 显示工单详情模态框
        const modal = new bootstrap.Modal(document.getElementById('workOrderDetailModal'));
        modal.show();

        // 更新模态框内容
        this.updateWorkOrderModal(order);
    }

    updateWorkOrderModal(order) {
        const equipment = this.equipmentStatus.get(order.equipment);
        const modal = document.getElementById('workOrderDetailModal');
        
        // 更新设备信息
        const equipmentInfo = modal.querySelector('.equipment-specs');
        equipmentInfo.innerHTML = `
            <div class="spec-item">
                <div class="spec-label">设备名称</div>
                <div class="spec-value">${equipment ? equipment.name : '未知设备'}</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">设备类型</div>
                <div class="spec-value">${equipment ? this.getEquipmentTypeText(equipment.type) : '未知'}</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">运行时长</div>
                <div class="spec-value">${equipment ? equipment.runningDays : 0}天</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">上次检修</div>
                <div class="spec-value">${equipment ? equipment.lastMaintenance : '无记录'}</div>
            </div>
        `;

        // 更新时间线
        const timeline = modal.querySelector('.maintenance-timeline');
        timeline.innerHTML = `
            <div class="timeline-item completed">
                <h6>创建工单</h6>
                <p>${order.startTime}</p>
                <small class="text-muted">由系统管理员创建</small>
            </div>
            ${order.tasks.map(task => `
                <div class="timeline-item ${task.status === 'completed' ? 'completed' : ''}">
                    <h6>${task.name}</h6>
                    <p>${task.status === 'completed' ? '已完成' : task.status === 'in-progress' ? '进行中' : '待执行'}</p>
                    <small class="text-muted">${this.getTaskStatusDescription(task)}</small>
                </div>
            `).join('')}
        `;
    }

    getEquipmentTypeText(type) {
        const typeMap = {
            'transformer': '变压器',
            'switchgear': '开关设备',
            'battery': '储能设备',
            'inverter': '逆变器',
            'cable': '电缆线路'
        };
        return typeMap[type] || type;
    }

    getTaskStatusDescription(task) {
        switch (task.status) {
            case 'completed':
                return '检修任务已完成';
            case 'in-progress':
                return '检修人员正在执行该任务';
            case 'pending':
                return '等待开始执行';
            default:
                return '';
        }
    }

    startRealTimeUpdates() {
        // 模拟实时更新
        setInterval(() => {
            this.simulateRealTimeUpdates();
        }, 5000);
    }

    simulateRealTimeUpdates() {
        // 模拟设备状态变化
        this.equipmentStatus.forEach(equipment => {
            if (Math.random() < 0.1) {
                equipment.status = this.getRandomStatus();
                this.updateEquipmentStatus();
            }
        });

        // 模拟工单进度更新
        this.workOrders.forEach(order => {
            if (order.status === 'in-progress') {
                order.tasks.forEach(task => {
                    if (task.status === 'in-progress' && Math.random() < 0.3) {
                        task.status = 'completed';
                        this.updateWorkOrdersList();
                    }
                });
            }
        });

        // 模拟无人机状态更新
        this.drones.forEach(drone => {
            if (drone.status === 'in-mission') {
                drone.battery = Math.max(0, drone.battery - 1);
                drone.flightTime = Math.max(0, drone.flightTime - 1);
                if (drone.currentTask) {
                    drone.currentTask.progress = Math.min(100, drone.currentTask.progress + 2);
                }
                this.updateDroneStatus();
            }
        });
    }

    getRandomStatus() {
        const statuses = ['normal', 'warning', 'danger'];
        return statuses[Math.floor(Math.random() * statuses.length)];
    }

    initializeEventListeners() {
        // 绑定事件监听器
        document.addEventListener('maintenance-status-updated', this.handleStatusUpdate.bind(this));
        
        // 绑定筛选器事件
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.addEventListener('click', () => this.handleFilter(chip));
        });
    }

    handleStatusUpdate(event) {
        const { equipmentId, status } = event.detail;
        if (this.equipmentStatus.has(equipmentId)) {
            this.equipmentStatus.get(equipmentId).status = status;
            this.updateEquipmentStatus();
        }
    }

    handleFilter(chip) {
        const type = chip.dataset.type;
        const value = chip.dataset.value;

        // 更新筛选器状态
        document.querySelectorAll(`[data-type="${type}"]`).forEach(c => {
            c.classList.remove('active');
        });
        chip.classList.add('active');

        // 应用筛选
        this.applyFilters();
    }

    applyFilters() {
        const activeFilters = Array.from(document.querySelectorAll('.filter-chip.active')).map(chip => ({
            type: chip.dataset.type,
            value: chip.dataset.value
        }));

        // 筛选工单
        const filteredOrders = Array.from(this.workOrders.values()).filter(order => {
            return activeFilters.every(filter => {
                switch (filter.type) {
                    case 'status':
                        return filter.value === 'all' || order.status === filter.value;
                    case 'priority':
                        return filter.value === 'all' || order.priority === filter.value;
                    case 'type':
                        return filter.value === 'all' || order.type === filter.value;
                    default:
                        return true;
                }
            });
        });

        // 更新显示
        this.updateFilteredWorkOrders(filteredOrders);
    }

    updateFilteredWorkOrders(orders) {
        const container = document.querySelector('.work-orders-list');
        if (!container) return;

        container.innerHTML = '';
        orders.forEach(order => {
            const card = this.createWorkOrderCard(order);
            container.appendChild(card);
        });
    }

    initializeCharts() {
        this.initializeHealthTrendChart();
        this.initializeTaskCompletionChart();
    }

    initializeHealthTrendChart() {
        const ctx = document.getElementById('equipment-health-chart');
        if (!ctx) return;

        // 模拟健康度趋势数据
        const data = {
            labels: Array.from({length: 7}, (_, i) => {
                const d = new Date();
                d.setDate(d.getDate() - (6 - i));
                return d.toLocaleDateString();
            }),
            datasets: [{
                label: '设备健康度',
                data: Array.from({length: 7}, () => Math.floor(Math.random() * 20 + 80)),
                borderColor: '#556ee6',
                backgroundColor: 'rgba(85, 110, 230, 0.1)',
                fill: true
            }]
        };

        new Chart(ctx, {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: value => value + '%'
                        }
                    }
                }
            }
        });
    }

    initializeTaskCompletionChart() {
        const ctx = document.getElementById('task-completion-chart');
        if (!ctx) return;

        // 模拟任务完成率数据
        const data = {
            labels: ['已完成', '进行中', '待处理'],
            datasets: [{
                data: [65, 20, 15],
                backgroundColor: ['#34c38f', '#556ee6', '#f1b44c']
            }]
        };

        new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    /**
     * 初始化各个组件
     */
    initializeComponents() {
        // 初始化检修看板
        this.initializeDashboard();
        
        // 初始化设备状态图表
        this.initializeEquipmentCharts();
        
        // 初始化检修日历
        this.initializeCalendar();
        
        // 初始化工单列表
        this.initializeWorkOrders();
    }

    /**
     * 初始化检修看板
     */
    initializeDashboard() {
        // 创建统计卡片
        const dashboardStats = [
            { label: '待处理工单', value: 12, trend: '+2', status: 'pending' },
            { label: '进行中工单', value: 8, trend: '-1', status: 'in-progress' },
            { label: '已完成工单', value: 45, trend: '+5', status: 'completed' },
            { label: '逾期工单', value: 3, trend: '+1', status: 'overdue' }
        ];

        const dashboardContainer = document.querySelector('.maintenance-dashboard');
        if (dashboardContainer) {
            dashboardContainer.innerHTML = dashboardStats.map(stat => `
                <div class="dashboard-card ${stat.status}">
                    <div class="card-label">${stat.label}</div>
                    <div class="card-value">${stat.value}</div>
                    <div class="trend ${parseInt(stat.trend) > 0 ? 'up' : 'down'}">
                        ${stat.trend}%
                        <i class="ri-arrow-${parseInt(stat.trend) > 0 ? 'up' : 'down'}-line"></i>
                    </div>
                </div>
            `).join('');
        }
    }

    /**
     * 初始化设备状态图表
     */
    initializeEquipmentCharts() {
        // 设备健康度环形图
        this.healthChart = new ApexCharts(document.querySelector("#equipment-health-chart"), {
            series: [75],
            chart: {
                type: 'radialBar',
                height: 200,
                sparkline: {
                    enabled: true
                }
            },
            plotOptions: {
                radialBar: {
                    startAngle: -90,
                    endAngle: 90,
                    track: {
                        background: "#333",
                        strokeWidth: '97%',
                        margin: 5
                    },
                    dataLabels: {
                        name: {
                            show: false
                        },
                        value: {
                            offsetY: -2,
                            fontSize: '22px',
                            formatter: function(val) {
                                return val + '%';
                            }
                        }
                    }
                }
            },
            grid: {
                padding: {
                    top: -10
                }
            },
            colors: ['#0dcaf0'],
            labels: ['设备健康度']
        });
        this.healthChart.render();

        // 检修任务完成率图表
        this.completionChart = new ApexCharts(document.querySelector("#task-completion-chart"), {
            series: [{
                name: '完成率',
                data: [65, 78, 82, 75, 85, 92, 88]
            }],
            chart: {
                type: 'area',
                height: 200,
                toolbar: {
                    show: false
                },
                sparkline: {
                    enabled: true
                }
            },
            stroke: {
                curve: 'smooth',
                width: 2
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shadeIntensity: 1,
                    opacityFrom: 0.7,
                    opacityTo: 0.3
                }
            },
            colors: ['#198754'],
            tooltip: {
                theme: 'dark',
                fixed: {
                    enabled: false
                },
                x: {
                    show: false
                },
                y: {
                    formatter: function(value) {
                        return value + '%';
                    }
                },
                marker: {
                    show: false
                }
            }
        });
        this.completionChart.render();
    }

    /**
     * 初始化检修日历
     */
    initializeCalendar() {
        const calendarContainer = document.querySelector('.maintenance-calendar');
        if (!calendarContainer) return;

        // 生成日历网格
        const today = new Date();
        const firstDay = new Date(today.getFullYear(), today.getMonth(), 1);
        const lastDay = new Date(today.getFullYear(), today.getMonth() + 1, 0);
        
        // 生成日历头部
        const monthNames = ['一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月'];
        calendarContainer.innerHTML = `
            <div class="calendar-header">
                <h5>${monthNames[today.getMonth()]} ${today.getFullYear()}</h5>
                <div class="btn-group">
                    <button class="btn btn-sm btn-dark" id="prev-month">
                        <i class="ri-arrow-left-s-line"></i>
                    </button>
                    <button class="btn btn-sm btn-dark" id="next-month">
                        <i class="ri-arrow-right-s-line"></i>
                    </button>
                </div>
            </div>
            <div class="calendar-grid">
                ${this._generateCalendarDays(firstDay, lastDay)}
            </div>
        `;
    }

    /**
     * 生成日历天数网格
     */
    _generateCalendarDays(firstDay, lastDay) {
        const days = [];
        const weekDays = ['日', '一', '二', '三', '四', '五', '六'];
        
        // 添加星期标题
        weekDays.forEach(day => {
            days.push(`<div class="calendar-day weekday">${day}</div>`);
        });

        // 添加空白天数
        for (let i = 0; i < firstDay.getDay(); i++) {
            days.push('<div class="calendar-day empty"></div>');
        }

        // 添加月份天数
        for (let i = 1; i <= lastDay.getDate(); i++) {
            const hasTask = Math.random() > 0.7; // 模拟某些日期有任务
            days.push(`
                <div class="calendar-day${hasTask ? ' has-task' : ''}" data-date="${i}">
                    <span>${i}</span>
                </div>
            `);
        }

        return days.join('');
    }

    /**
     * 初始化工单列表
     */
    initializeWorkOrders() {
        const workOrdersContainer = document.querySelector('.work-orders-list');
        if (!workOrdersContainer) return;

        // 模拟工单数据
        const workOrders = [
            {
                id: 'WO-20240405-001',
                title: '主变压器#1例行检修',
                type: '例行检修',
                status: '进行中',
                priority: '普通',
                startTime: '2024-04-05 09:00',
                endTime: '2024-04-05 17:00',
                progress: 45
            },
            {
                id: 'WO-20240405-002',
                title: '储能系统#2故障检修',
                type: '故障检修',
                status: '待执行',
                priority: '紧急',
                startTime: '2024-04-05 14:00',
                endTime: '2024-04-05 18:00',
                progress: 0
            }
            // ... 更多工单数据
        ];

        // 渲染工单列表
        workOrdersContainer.innerHTML = workOrders.map(order => this._generateWorkOrderHtml(order)).join('');
    }

    /**
     * 生成工单HTML
     */
    _generateWorkOrderHtml(order) {
        return `
            <div class="maintenance-order ${order.priority}" data-id="${order.id}">
                <div class="order-header">
                    <div class="order-title">${order.title}</div>
                    <span class="badge bg-${this._getStatusBadgeColor(order.status)}">${this._getStatusLabel(order.status)}</span>
                </div>
                <div class="order-meta">
                    <div class="meta-item">
                        <i class="ri-tools-line"></i>
                        <span>${this.config.maintenanceTypes[order.type]}</span>
                    </div>
                    <div class="meta-item">
                        <i class="ri-calendar-line"></i>
                        <span>${order.startTime}</span>
                    </div>
                    <div class="meta-item">
                        <i class="ri-user-line"></i>
                        <span>${order.assignee}</span>
                    </div>
                </div>
                <div class="order-actions mt-3">
                    <button class="btn btn-sm btn-primary view-order" data-id="${order.id}">
                        <i class="ri-eye-line me-1"></i>查看详情
                    </button>
                    ${order.status === 'pending' ? `
                        <button class="btn btn-sm btn-success start-order" data-id="${order.id}">
                            <i class="ri-play-line me-1"></i>开始执行
                        </button>
                    ` : ''}
                    ${order.status === 'in-progress' ? `
                        <button class="btn btn-sm btn-info complete-order" data-id="${order.id}">
                            <i class="ri-check-line me-1"></i>完成工单
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    /**
     * 获取状态标签颜色
     */
    _getStatusBadgeColor(status) {
        const statusColors = {
            'pending': 'warning',
            'in-progress': 'info',
            'completed': 'success',
            'cancelled': 'danger'
        };
        return statusColors[status] || 'secondary';
    }

    /**
     * 获取状态标签文本
     */
    _getStatusLabel(status) {
        const statusLabels = {
            'pending': '待处理',
            'in-progress': '进行中',
            'completed': '已完成',
            'cancelled': '已取消'
        };
        return statusLabels[status] || status;
    }

    /**
     * 绑定事件处理
     */
    bindEvents() {
        // 绑定工单相关事件
        document.addEventListener('click', (e) => {
            if (e.target.closest('.view-order')) {
                const orderId = e.target.closest('.view-order').dataset.id;
                this.showOrderDetails(orderId);
            } else if (e.target.closest('.start-order')) {
                const orderId = e.target.closest('.start-order').dataset.id;
                this.startOrder(orderId);
            } else if (e.target.closest('.complete-order')) {
                const orderId = e.target.closest('.complete-order').dataset.id;
                this.completeOrder(orderId);
            }
        });

        // 绑定日历事件
        document.addEventListener('click', (e) => {
            if (e.target.closest('.calendar-day')) {
                const date = e.target.closest('.calendar-day').dataset.date;
                if (date) {
                    this.showDateTasks(date);
                }
            } else if (e.target.closest('#prev-month')) {
                this.navigateMonth(-1);
            } else if (e.target.closest('#next-month')) {
                this.navigateMonth(1);
            }
        });

        // 绑定筛选事件
        document.addEventListener('click', (e) => {
            if (e.target.closest('.filter-chip')) {
                const filter = e.target.closest('.filter-chip');
                const type = filter.dataset.type;
                const value = filter.dataset.value;
                this.applyFilter(type, value);
            }
        });

        // 新建工单按钮
        document.getElementById('newWorkOrder')?.addEventListener('click', () => {
            const modal = new bootstrap.Modal(document.getElementById('newWorkOrderModal'));
            modal.show();
        });

        // 新建巡检任务按钮
        document.getElementById('newDroneTask')?.addEventListener('click', () => {
            const modal = new bootstrap.Modal(document.getElementById('newDroneTaskModal'));
            modal.show();
        });
    }

    /**
     * 显示工单详情
     */
    showOrderDetails(orderId) {
        // 实现工单详情显示逻辑
        console.log('显示工单详情:', orderId);
    }

    /**
     * 开始执行工单
     */
    startOrder(orderId) {
        // 实现开始执行工单逻辑
        console.log('开始执行工单:', orderId);
    }

    /**
     * 完成工单
     */
    completeOrder(orderId) {
        // 实现完成工单逻辑
        console.log('完成工单:', orderId);
    }

    /**
     * 显示日期任务
     */
    showDateTasks(date) {
        // 实现显示日期任务逻辑
        console.log('显示日期任务:', date);
    }

    /**
     * 切换月份
     */
    navigateMonth(delta) {
        // 实现月份切换逻辑
        console.log('切换月份:', delta);
    }

    /**
     * 应用筛选
     */
    applyFilter(type, value) {
        // 实现筛选逻辑
        console.log('应用筛选:', type, value);
    }

    /**
     * 加载看板数据
     */
    loadDashboardData() {
        // 实现加载看板数据逻辑
        console.log('加载看板数据');
    }

    /**
     * 初始化无人机功能
     */
    initializeDroneFeatures() {
        this.initializeDroneTasks();
        this.initializeFleetStatus();
        this.initializeFleetMonitoring();
        this.bindDroneEvents();
        this.startDroneStatusUpdates();
    }

    /**
     * 初始化无人机任务列表
     */
    initializeDroneTasks() {
        const taskList = document.querySelector('.drone-tasks .task-list');
        if (!taskList) return;

        // 模拟任务数据
        const tasks = [
            {
                id: 'TASK001',
                area: 'area1',
                type: 'routine',
                startTime: '2024-04-01 10:00',
                status: 'pending',
                findings: 0
            },
            {
                id: 'TASK002',
                area: 'area2',
                type: 'thermal',
                startTime: '2024-04-01 14:30',
                status: 'in-progress',
                findings: 2
            }
        ];

        taskList.innerHTML = tasks.map(task => this._generateTaskHtml(task)).join('');
    }

    /**
     * 生成任务HTML
     */
    _generateTaskHtml(task) {
        const type = this.droneConfig.inspectionTypes[task.type];
        const area = this.droneConfig.areas[task.area];
        
        return `
            <div class="task-item" data-task-id="${task.id}">
                <div class="task-header">
                    <div class="task-area">
                        <i class="${type.icon} me-1"></i>
                        ${area.label}
                    </div>
                    <span class="badge bg-${this._getStatusBadgeColor(task.status)}">
                        ${this._getStatusLabel(task.status)}
                    </span>
                </div>
                <div class="task-info">
                    <div class="meta-item">
                        <i class="ri-time-line me-1"></i>
                        ${task.startTime}
                    </div>
                    ${task.findings > 0 ? `
                        <div class="meta-item text-warning">
                            <i class="ri-error-warning-line me-1"></i>
                            发现 ${task.findings} 个问题
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    /**
     * 绑定无人机相关事件
     */
    bindDroneEvents() {
        // 查看任务详情
        document.querySelector('.drone-tasks')?.addEventListener('click', (e) => {
            const taskItem = e.target.closest('.task-item');
            if (taskItem) {
                this.showTaskDetails(taskItem.dataset.taskId);
            }
        });

        // 拍照按钮
        document.getElementById('takePhoto')?.addEventListener('click', () => {
            this.takePhoto();
        });

        // 录像按钮
        document.getElementById('startRecord')?.addEventListener('click', (e) => {
            const btn = e.target.closest('button');
            if (btn.classList.contains('recording')) {
                this.stopRecording();
                btn.classList.remove('recording');
                btn.innerHTML = '<i class="ri-record-circle-line me-1"></i>录像';
            } else {
                this.startRecording();
                btn.classList.add('recording');
                btn.innerHTML = '<i class="ri-stop-circle-line me-1"></i>停止';
            }
        });

        // 导出报告
        document.getElementById('exportReport')?.addEventListener('click', () => {
            this.exportInspectionReport();
        });
    }

    /**
     * 开始无人机状态更新
     */
    startDroneStatusUpdates() {
        // 模拟实时更新无人机状态
        setInterval(() => {
            if (this.droneStatus.status === 'flying') {
                // 更新电量
                this.droneStatus.battery = Math.max(0, this.droneStatus.battery - 0.1);
                // 更新剩余飞行时间
                this.droneStatus.flightTime = Math.floor(this.droneStatus.battery * 0.3);
                
                // 更新状态显示
                this.updateDroneStatusDisplay();
            }
        }, 1000);
    }

    /**
     * 更新无人机状态显示
     */
    updateDroneStatusDisplay() {
        const batteryEl = document.querySelector('.drone-battery span');
        const statusEl = document.querySelector('.drone-info .value.text-success');
        const flightTimeEl = document.querySelector('.drone-info .value:last-child');

        if (batteryEl) batteryEl.textContent = `${Math.floor(this.droneStatus.battery)}%`;
        if (statusEl) statusEl.textContent = this._getDroneStatusLabel(this.droneStatus.status);
        if (flightTimeEl) flightTimeEl.textContent = `${this.droneStatus.flightTime}分钟`;
    }

    /**
     * 创建无人机巡检任务
     */
    createDroneTask() {
        const area = document.getElementById('inspectionArea').value;
        const type = document.getElementById('inspectionType').value;
        const startTime = document.getElementById('taskStartTime').value;
        const duration = document.getElementById('estimatedDuration').value;
        const height = document.getElementById('flightHeight').value;
        const speed = document.getElementById('flightSpeed').value;

        // 验证输入
        if (!area || !type || !startTime || !duration || !height || !speed) {
            alert('请填写完整的任务信息');
            return;
        }

        // 创建任务
        const task = {
            id: `TASK${Date.now()}`,
            area,
            type,
            startTime,
            status: 'pending',
            duration,
            height,
            speed,
            findings: 0
        };

        // 添加到任务列表
        const taskList = document.querySelector('.drone-tasks .task-list');
        if (taskList) {
            taskList.insertAdjacentHTML('afterbegin', this._generateTaskHtml(task));
        }

        // 关闭模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('newDroneTaskModal'));
        modal.hide();
    }

    /**
     * 显示任务详情
     */
    showTaskDetails(taskId) {
        // 获取任务数据
        const task = this._getTaskById(taskId);
        if (!task) return;

        // 更新模态框内容
        document.getElementById('taskId').textContent = task.id;
        document.getElementById('taskStatus').textContent = this._getStatusLabel(task.status);
        
        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('droneTaskDetailModal'));
        modal.show();
    }

    /**
     * 拍照
     */
    takePhoto() {
        // 实现拍照功能
        console.log('拍照');
        // 这里可以调用无人机API进行拍照
    }

    /**
     * 开始录像
     */
    startRecording() {
        // 实现开始录像功能
        console.log('开始录像');
        // 这里可以调用无人机API开始录像
    }

    /**
     * 停止录像
     */
    stopRecording() {
        // 实现停止录像功能
        console.log('停止录像');
        // 这里可以调用无人机API停止录像
    }

    /**
     * 导出巡检报告
     */
    exportInspectionReport() {
        // 实现导出报告功能
        console.log('导出报告');
        // 这里可以生成PDF报告
    }

    /**
     * 获取无人机状态标签
     */
    _getDroneStatusLabel(status) {
        const labels = {
            standby: '待命中',
            flying: '飞行中',
            charging: '充电中',
            maintenance: '维护中'
        };
        return labels[status] || status;
    }

    /**
     * 根据ID获取任务
     */
    _getTaskById(taskId) {
        // 实现任务查询
        // 这里需要连接后端API获取任务详情
        return null;
    }

    /**
     * 初始化无人机集群状态显示
     */
    initializeFleetStatus() {
        const fleetStatusContainer = document.querySelector('.drone-fleet-status');
        if (!fleetStatusContainer) return;

        const activeFleet = this.droneFleetConfig.fleets[this.fleetStatus.activeFleet];
        
        fleetStatusContainer.innerHTML = `
            <div class="fleet-overview">
                <div class="fleet-header">
                    <h6>${activeFleet.name}</h6>
                    <div class="fleet-stats">
                        <span class="badge bg-success">${this.fleetStatus.availableDrones} 可用</span>
                        <span class="badge bg-warning">${this.fleetStatus.chargingDrones} 充电中</span>
                        <span class="badge bg-danger">${this.fleetStatus.maintenanceDrones} 维护中</span>
                    </div>
                </div>
                <div class="drone-list">
                    ${activeFleet.drones.map(drone => this._generateDroneItemHtml(drone)).join('')}
                </div>
                <div class="fleet-capabilities">
                    <h6>集群能力</h6>
                    <div class="capability-tags">
                        ${activeFleet.capabilities.map(cap => `
                            <span class="capability-tag">
                                <i class="ri-checkbox-circle-line"></i>
                                ${this.droneFleetConfig.sensorTypes[cap].label}
                            </span>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * 生成无人机项HTML
     */
    _generateDroneItemHtml(drone) {
        return `
            <div class="drone-item ${drone.status}" data-drone-id="${drone.id}">
                <div class="drone-info">
                    <div class="drone-header">
                        <span class="drone-id">${drone.id}</span>
                        <span class="drone-role ${drone.role}">${drone.role}</span>
                    </div>
                    <div class="drone-model">${drone.model}</div>
                    <div class="drone-status">
                        <div class="battery-indicator">
                            <i class="ri-battery-2-line"></i>
                            <span>${drone.battery}%</span>
                        </div>
                        <span class="status-badge">${this._getDroneStatusLabel(drone.status)}</span>
                    </div>
                </div>
                <div class="drone-actions">
                    <button class="btn btn-sm btn-primary" onclick="maintenanceManagement.controlDrone('${drone.id}', 'takeoff')">
                        <i class="ri-flight-takeoff-line"></i>
                    </button>
                    <button class="btn btn-sm btn-warning" onclick="maintenanceManagement.controlDrone('${drone.id}', 'return')">
                        <i class="ri-home-line"></i>
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * 初始化集群监控
     */
    initializeFleetMonitoring() {
        const monitoringContainer = document.querySelector('.fleet-monitoring');
        if (!monitoringContainer) return;

        monitoringContainer.innerHTML = `
            <div class="monitoring-header">
                <h6>集群监控</h6>
                <div class="weather-info">
                    <span><i class="ri-temp-hot-line"></i> ${this.fleetStatus.weatherConditions.temperature}°C</span>
                    <span><i class="ri-windy-line"></i> ${this.fleetStatus.weatherConditions.windSpeed}m/s</span>
                    <span><i class="ri-sun-line"></i> ${this.fleetStatus.weatherConditions.visibility}</span>
                </div>
            </div>
            <div class="monitoring-map" id="fleetMap">
                <!-- 地图将通过JS动态加载 -->
            </div>
            <div class="flight-restrictions">
                <h6>飞行限制</h6>
                <div class="restriction-items">
                    <div class="restriction-item">
                        <span class="label">最大高度</span>
                        <span class="value">${this.fleetStatus.flightRestrictions.maxHeight}m</span>
                    </div>
                    <div class="restriction-item">
                        <span class="label">最大速度</span>
                        <span class="value">${this.fleetStatus.flightRestrictions.maxSpeed}m/s</span>
                    </div>
                </div>
            </div>
        `;

        // 初始化地图
        this.initializeFleetMap();
    }

    /**
     * 初始化集群地图
     */
    initializeFleetMap() {
        // 这里应该实现地图初始化逻辑
        // 使用地图API（如高德地图、百度地图等）
        console.log('初始化集群地图');
    }

    /**
     * 控制无人机
     */
    controlDrone(droneId, action) {
        console.log(`控制无人机 ${droneId} 执行 ${action}`);
        // 这里应该实现实际的无人机控制逻辑
    }

    /**
     * 创建集群巡检任务
     */
    createFleetTask() {
        const fleet = document.getElementById('fleetSelect').value;
        const mode = document.getElementById('inspectionMode').value;
        const pattern = document.getElementById('inspectionPattern').value;
        const area = document.getElementById('inspectionArea').value;

        // 验证输入
        if (!fleet || !mode || !pattern || !area) {
            alert('请填写完整的任务信息');
            return;
        }

        // 创建任务
        const task = {
            id: `FLEET-TASK-${Date.now()}`,
            fleet,
            mode,
            pattern,
            area,
            status: 'pending',
            startTime: new Date().toISOString(),
            drones: this.droneFleetConfig.fleets[fleet].drones
                .filter(drone => drone.status === 'standby')
                .map(drone => drone.id)
        };

        // 添加到任务列表
        this.addFleetTask(task);

        // 关闭模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('newFleetTaskModal'));
        modal.hide();
    }

    /**
     * 添加集群任务
     */
    addFleetTask(task) {
        const taskList = document.querySelector('.fleet-tasks .task-list');
        if (!taskList) return;

        const taskHtml = `
            <div class="fleet-task-item" data-task-id="${task.id}">
                <div class="task-header">
                    <div class="task-info">
                        <h6>${this.droneFleetConfig.fleets[task.fleet].name}</h6>
                        <span class="badge bg-info">
                            ${this.droneFleetConfig.inspectionModes[task.mode].label}
                        </span>
                    </div>
                    <div class="task-status">
                        <span class="badge bg-warning">待执行</span>
                    </div>
                </div>
                <div class="task-details">
                    <div class="detail-item">
                        <i class="ri-map-pin-line"></i>
                        <span>${this.droneConfig.areas[task.area].label}</span>
                    </div>
                    <div class="detail-item">
                        <i class="ri-group-line"></i>
                        <span>${task.drones.length} 架无人机</span>
                    </div>
                    <div class="detail-item">
                        <i class="ri-flight-path-line"></i>
                        <span>${this.droneFleetConfig.inspectionPatterns[task.pattern].label}</span>
                    </div>
                </div>
                <div class="task-actions">
                    <button class="btn btn-sm btn-primary" onclick="maintenanceManagement.startFleetTask('${task.id}')">
                        开始任务
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="maintenanceManagement.cancelFleetTask('${task.id}')">
                        取消任务
                    </button>
                </div>
            </div>
        `;

        taskList.insertAdjacentHTML('afterbegin', taskHtml);
    }

    /**
     * 开始集群任务
     */
    startFleetTask(taskId) {
        console.log('开始集群任务:', taskId);
        // 实现任务启动逻辑
    }

    /**
     * 取消集群任务
     */
    cancelFleetTask(taskId) {
        console.log('取消集群任务:', taskId);
        // 实现任务取消逻辑
    }
}

// 导出模块
window.MaintenanceManagement = MaintenanceManagement;

// 初始化设备检修管理模块
document.addEventListener('DOMContentLoaded', () => {
    window.maintenanceManagement = new MaintenanceManagement();
}); 