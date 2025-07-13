class ChargingManagement {
    constructor() {
        this.stations = new Map();
        this.initialize();
    }

    async initialize() {
        try {
            await this.loadStationData();
            this.initializeEventListeners();
            // 使用 SweetAlert2 显示初始化成功提示
            Swal.fire({
                title: '系统就绪',
                text: '充电管理系统初始化完成',
                icon: 'success',
                toast: true,
                position: 'top-end',
                showConfirmButton: false,
                timer: 3000,
                timerProgressBar: true,
                background: '#2a3042',
                color: '#fff'
            });
        } catch (error) {
            console.error('充电管理系统初始化失败:', error);
            // 使用 SweetAlert2 显示错误提示
            Swal.fire({
                title: '初始化失败',
                text: '系统启动时发生错误，请刷新页面重试',
                icon: 'error',
                confirmButtonText: '确定',
                background: '#2a3042',
                color: '#fff'
            });
        }
    }

    async loadStationData() {
        try {
            // 模拟加载数据
            const stations = [
                {
                    id: 'CS001',
                    name: '成都高新区A站',
                    location: [104.06, 30.67],
                    status: '正常',
                    type: '快充站',
                    power: '120kW',
                    voltage: '750V',
                    current: '160A',
                    totalPorts: 12,
                    availablePorts: 10,
                    chargingPorts: 2,
                    faultPorts: 0,
                    todayCharges: 45,
                    todayEnergy: 1250.5,
                    todayIncome: 1580.75
                },
                // ... 其他充电站数据
            ];

            // 存储数据
            stations.forEach(station => {
                this.stations.set(station.id, station);
            });

            // 更新概览数据
            this.updateOverviewData();

            return true;
        } catch (error) {
            console.error('加载充电站数据失败:', error);
            // 使用 SweetAlert2 显示错误提示
            Swal.fire({
                title: '数据加载失败',
                text: '无法加载充电站数据，请检查网络连接',
                icon: 'error',
                confirmButtonText: '重试',
                background: '#2a3042',
                color: '#fff'
            }).then((result) => {
                if (result.isConfirmed) {
                    this.loadStationData(); // 重试加载数据
                }
            });
            return false;
        }
    }

    updateOverviewData() {
        try {
            // 计算总数据
            let totalStations = this.stations.size;
            let onlineStations = Array.from(this.stations.values()).filter(s => s.status === '正常').length;
            let totalCharges = Array.from(this.stations.values()).reduce((sum, s) => sum + s.todayCharges, 0);
            let totalIncome = Array.from(this.stations.values()).reduce((sum, s) => sum + s.todayIncome, 0);

            // 更新DOM
            const overviewElements = {
                totalStations: document.querySelector('[data-overview="total-stations"]'),
                onlineRate: document.querySelector('[data-overview="online-rate"]'),
                todayCharges: document.querySelector('[data-overview="today-charges"]'),
                todayIncome: document.querySelector('[data-overview="today-income"]')
            };

            if (overviewElements.totalStations) {
                overviewElements.totalStations.textContent = totalStations;
            }
            if (overviewElements.onlineRate) {
                overviewElements.onlineRate.textContent = ((onlineStations / totalStations) * 100).toFixed(1) + '%';
            }
            if (overviewElements.todayCharges) {
                overviewElements.todayCharges.textContent = totalCharges;
            }
            if (overviewElements.todayIncome) {
                overviewElements.todayIncome.textContent = '¥' + totalIncome.toFixed(2);
            }

        } catch (error) {
            console.error('更新概览数据失败:', error);
            // 使用 SweetAlert2 显示错误提示
            Swal.fire({
                title: '数据更新失败',
                text: '更新概览数据时发生错误',
                icon: 'warning',
                toast: true,
                position: 'top-end',
                showConfirmButton: false,
                timer: 3000,
                background: '#2a3042',
                color: '#fff'
            });
        }
    }

    initializeEventListeners() {
        // 添加事件监听器
        document.addEventListener('charging-station-updated', this.handleStationUpdate.bind(this));
    }

    handleStationUpdate(event) {
        const { stationId, data } = event.detail;
        if (this.stations.has(stationId)) {
            this.stations.set(stationId, { ...this.stations.get(stationId), ...data });
            this.updateOverviewData();
            
            // 使用 SweetAlert2 显示更新成功提示
            Swal.fire({
                title: '数据已更新',
                text: `充电站 ${stationId} 数据已更新`,
                icon: 'success',
                toast: true,
                position: 'top-end',
                showConfirmButton: false,
                timer: 2000,
                background: '#2a3042',
                color: '#fff'
            });
        }
    }

    // 添加应急模式切换方法
    toggleEmergencyMode(enabled) {
        Swal.fire({
            title: enabled ? '确认启动应急模式？' : '确认关闭应急模式？',
            text: enabled ? '这将限制所有充电桩的最大功率输出' : '这将恢复充电桩的正常运行模式',
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: '确认',
            cancelButtonText: '取消',
            background: '#2a3042',
            color: '#fff'
        }).then((result) => {
            if (result.isConfirmed) {
                // 执行应急模式切换逻辑
                Swal.fire({
                    title: enabled ? '应急模式已启动' : '应急模式已关闭',
                    icon: 'success',
                    toast: true,
                    position: 'top-end',
                    showConfirmButton: false,
                    timer: 2000,
                    background: '#2a3042',
                    color: '#fff'
                });
            }
        });
    }
}

// 导出类
window.ChargingManagement = ChargingManagement; 