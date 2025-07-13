// 无人机巡检系统主要功能
class DroneInspectionSystem {
    constructor() {
        this.map = null;
        this.droneMarker = null;
        this.inspectionPath = [];
        this.currentTask = null;
        this.isPaused = false;
        this.currentMode = 'inspection';
        this.eventListenersInitialized = false; // 添加标志位

        // 基准值定义
        this.baseValues = {
            droneHeight: 50.5,
            droneSpeed: 5.2,
            batteryLevel: 75,
            signalStrength: -65,  // 基准信号强度为 -65dBm
            surfaceTemp: 32.5,    // 表面温度 (°C)
            tempDiff: 2.3,        // 温差异常 (°C)
            hotSpotCount: 0,      // 热点数量
            surfaceDamage: 0.5,   // 表面损伤率 (%)
            contamination: 3.2,    // 污渍覆盖率 (%)
            deformation: 0.2,     // 形变量 (mm)
            vibration: 0.15,      // 振动强度 (g)
            noise: 65,            // 噪声水平 (dB)
            resonance: 12.5,      // 共振频率 (Hz)
            corona: 15,           // 电晕放电 (dB)
            insulation: 1,        // 绝缘状态 (0-故障, 1-正常)
            bladeAngle: 15,       // 叶片角度 (°)
            panelTilt: 32,        // 光伏板倾角 (°)
            envTemp: 25.8,        // 环境温度 (°C)
            tempGradient: 0.6,    // 温度梯度 (°C/100m)
            humidity: 45,         // 相对湿度 (%)
            dewPoint: 13.2,       // 露点温度 (°C)
            windSpeed: 3.2,       // 风速 (m/s)
            windDirection: 225,    // 风向 (度)
            windShear: 0.03,      // 风切变 (s⁻¹)
            turbulence: 0.15,     // 湍流强度
            visibility: 12,       // 能见度 (km)
            pressure: 1013.25,    // 气压 (hPa)
            airDensity: 1.225,    // 空气密度 (kg/m³)
            solarRadiation: 800,   // 太阳辐射 (W/m²)
            cloudCover: 25,        // 云量 (%)
            clearnessIndex: 0.75,  // 晴空指数
            pm25: 35,             // PM2.5 (µg/m³)
            dust: 0.15            // 粉尘浓度 (mg/m³)
        };

        // 波动范围定义
        this.fluctuationRanges = {
            droneHeight: 2,       // ±2%
            droneSpeed: 5,        // ±5%
            batteryLevel: 0.2,    // ±0.2%
            signalStrength: 5,    // ±5dBm
            surfaceTemp: 1,       // ±1%
            tempDiff: 0.5,        // ±0.5°C
            hotSpotCount: 1,      // ±1个
            surfaceDamage: 0.1,   // ±0.1%
            contamination: 0.5,    // ±0.5%
            deformation: 0.1,     // ±0.1mm
            vibration: 10,         // ±10%
            noise: 3,             // ±3dB
            resonance: 5,         // ±5%
            corona: 2,            // ±2dB
            insulation: 0,        // 不波动
            bladeAngle: 2,        // ±2°
            panelTilt: 0.5,       // ±0.5°
            envTemp: 0.5,         // ±0.5°C
            tempGradient: 0.1,    // ±0.1°C/100m
            humidity: 2,          // ±2%
            dewPoint: 0.5,         // ±0.5°C
            windSpeed: 0.5,        // ±0.5m/s
            windDirection: 10,     // ±10°
            windShear: 0.01,       // ±0.01s⁻¹
            turbulence: 0.05,      // ±0.05
            visibility: 0.5,       // ±0.5km
            pressure: 0.25,        // ±0.25hPa
            airDensity: 0.005,     // ±0.005kg/m³
            solarRadiation: 50,     // ±50W/m²
            cloudCover: 5,          // ±5%
            clearnessIndex: 0.05,   // ±0.05
            pm25: 5,               // ±5µg/m³
            dust: 0.02            // ±0.02mg/m³
        };

        // 阈值定义
        this.thresholds = {
            surfaceTemp: { low: 10, high: 40 },
            tempDiff: { low: 0, high: 5 },
            hotSpotCount: { low: 0, high: 1 },
            surfaceDamage: { low: 0, high: 1 },
            contamination: { low: 0, high: 3 },
            deformation: { low: 0, high: 0.5 },
            vibration: { low: 0.05, high: 0.2 },
            noise: { low: 50, high: 75 },
            resonance: { low: 10, high: 15 },
            corona: { low: 0, high: 20 },
            bladeAngle: { low: 5, high: 25 },
            panelTilt: { low: 25, high: 40 }
        };

        // 气象数据阈值
        this.weatherThresholds = {
            envTemp: { low: 10, high: 35 },
            tempGradient: { low: 0.3, high: 1.0 },
            humidity: { low: 20, high: 80 },
            dewPoint: { low: 0, high: 20 },
            windSpeed: { low: 0, high: 8 },
            windShear: { low: 0, high: 0.05 },
            turbulence: { low: 0, high: 0.2 },
            visibility: { low: 5, high: 15 },
            pressure: { low: 1000, high: 1020 },
            airDensity: { low: 1.1, high: 1.3 },
            solarRadiation: { low: 200, high: 1000 },
            cloudCover: { low: 0, high: 50 },
            clearnessIndex: { low: 0.5, high: 0.9 },
            pm25: { low: 0, high: 50 },
            dust: { low: 0, high: 0.2 }
        };

        this.initializeSystem();
        this.initializeFullscreen();
        this.initializeCharts();
        this.addNotificationStyles();
    }

    // 初始化系统
    initializeSystem() {
        try {
            // 只初始化必要的功能
            this.simulateRealTimeData(); // 优先启动实时数据模拟
            
            // 其他初始化可以在需要时进行
            if (document.getElementById('map')) {
                this.initializeMap();
            }
            
            // 只初始化一次事件监听器
            if (!this.eventListenersInitialized) {
                this.initializeEventListeners();
                this.eventListenersInitialized = true;
            }
            
        } catch (error) {
            console.error('Error initializing system:', error);
        }
    }

    // 初始化地图
    initializeMap() {
        try {
            // 四川会理县某光伏电站的实际坐标
            const powerPlantLocation = {
                center: [26.4296, 102.2485],  // 会理县光伏电站实际位置
                zoom: 15  // 调整缩放级别以更好地显示电站范围
            };

            // 初始化地图
            this.map = L.map('map').setView(powerPlantLocation.center, powerPlantLocation.zoom);
            
            // 添加卫星图层
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            }).addTo(this.map);

            // 添加发电厂边界（实际光伏电站边界）
            const powerPlantBoundary = [
                [26.4336, 102.2435],
                [26.4336, 102.2535],
                [26.4256, 102.2535],
                [26.4256, 102.2435],
                [26.4336, 102.2435]
            ];

            L.polygon(powerPlantBoundary, {
                color: '#2196F3',
                weight: 2,
                fillColor: '#2196F3',
                fillOpacity: 0.1
            }).addTo(this.map);

            // 添加光伏区域标记（根据实际光伏板排布）
            const facilities = [
                // A区光伏板组
                { type: 'solar', lat: 26.4316, lng: 102.2455, name: '光伏区A-1' },
                { type: 'solar', lat: 26.4316, lng: 102.2475, name: '光伏区A-2' },
                { type: 'solar', lat: 26.4316, lng: 102.2495, name: '光伏区A-3' },
                // B区光伏板组
                { type: 'solar', lat: 26.4296, lng: 102.2455, name: '光伏区B-1' },
                { type: 'solar', lat: 26.4296, lng: 102.2475, name: '光伏区B-2' },
                { type: 'solar', lat: 26.4296, lng: 102.2495, name: '光伏区B-3' },
                // C区光伏板组
                { type: 'solar', lat: 26.4276, lng: 102.2455, name: '光伏区C-1' },
                { type: 'solar', lat: 26.4276, lng: 102.2475, name: '光伏区C-2' },
                { type: 'solar', lat: 26.4276, lng: 102.2495, name: '光伏区C-3' },
                // 集电站
                { type: 'substation', lat: 26.4266, lng: 102.2515, name: '集电站' }
            ];

            facilities.forEach(facility => {
                const icon = L.divIcon({
                    className: 'facility-marker',
                    html: `<i class="bi bi-${facility.type === 'solar' ? 'sun' : 'lightning-charge'} text-primary"></i>`,
                    iconSize: [30, 30]
                });

                L.marker([facility.lat, facility.lng], { icon })
                    .bindPopup(facility.name)
                    .addTo(this.map);
            });

            // 添加无人机标记
            const droneIcon = L.divIcon({
                className: 'drone-marker',
                html: '<i class="bi bi-airplane text-danger"></i>',
                iconSize: [30, 30]
            });

            this.droneMarker = L.marker(powerPlantLocation.center, {
                icon: droneIcon,
                zIndexOffset: 1000
            }).addTo(this.map);

            // 初始化巡检路径图层
            this.inspectionPathLayer = L.layerGroup().addTo(this.map);
            
            // 设置默认巡检路径
            this.setDefaultInspectionPath();
        } catch (error) {
            console.error('Error initializing map:', error);
        }
    }

    // 设置默认巡检路径
    setDefaultInspectionPath() {
        try {
            // 清除现有路径
            this.inspectionPathLayer.clearLayers();

            // 定义巡检路径点（按照光伏区域排布设计路径）
            const pathPoints = [
                [26.4316, 102.2455], // A-1
                [26.4316, 102.2475], // A-2
                [26.4316, 102.2495], // A-3
                [26.4296, 102.2495], // B-3
                [26.4296, 102.2475], // B-2
                [26.4296, 102.2455], // B-1
                [26.4276, 102.2455], // C-1
                [26.4276, 102.2475], // C-2
                [26.4276, 102.2495], // C-3
                [26.4266, 102.2515], // 集电站
                [26.4316, 102.2455]  // 返回起点
            ];

            // 绘制巡检路径线
            const pathLine = L.polyline(pathPoints, {
                color: '#FF4081',
                weight: 4,
                opacity: 0.8,
                dashArray: '15, 15',
                animate: true
            }).addTo(this.inspectionPathLayer);

            // 添加方向箭头
            const decorator = L.polylineDecorator(pathLine, {
                patterns: [
                    {
                        offset: '5%',
                        repeat: '15%',
                        symbol: L.Symbol.arrowHead({
                            pixelSize: 20,
                            polygon: false,
                            pathOptions: {
                                color: '#FF4081',
                                fillOpacity: 1,
                                weight: 3
                            }
                        })
                    }
                ]
            }).addTo(this.inspectionPathLayer);

            // 存储路径点供动画使用
            this.inspectionPath = pathPoints;
            
            // 开始无人机路径动画
            this.startDroneAnimation();
        } catch (error) {
            console.error('Error setting inspection path:', error);
        }
    }

    // 开始无人机路径动画
    startDroneAnimation() {
        if (!this.inspectionPath || this.inspectionPath.length < 2) return;

        let currentPointIndex = 0;
        const animationDuration = 3000; // 每段路径的动画时间（毫秒）

        const animateToNextPoint = () => {
            const currentPoint = this.inspectionPath[currentPointIndex];
            const nextPoint = this.inspectionPath[(currentPointIndex + 1) % this.inspectionPath.length];

            // 计算朝向角度
            const angle = this.calculateBearing(currentPoint, nextPoint);

            // 更新无人机图标的旋转
            const droneIcon = this.droneMarker.getElement();
            if (droneIcon) {
                droneIcon.style.transform = `rotate(${angle}deg)`;
            }

            // 使用动画移动无人机
            this.animateDroneMovement(currentPoint, nextPoint, animationDuration, () => {
                currentPointIndex = (currentPointIndex + 1) % this.inspectionPath.length;
                setTimeout(animateToNextPoint, 1000); // 在每个点停留1秒
            });
        };

        // 开始动画
        animateToNextPoint();
    }

    // 计算两点之间的方位角
    calculateBearing(start, end) {
        const startLat = start[0] * Math.PI / 180;
        const startLng = start[1] * Math.PI / 180;
        const endLat = end[0] * Math.PI / 180;
        const endLng = end[1] * Math.PI / 180;

        const dLng = endLng - startLng;

        const y = Math.sin(dLng) * Math.cos(endLat);
        const x = Math.cos(startLat) * Math.sin(endLat) -
                 Math.sin(startLat) * Math.cos(endLat) * Math.cos(dLng);

        let bearing = Math.atan2(y, x) * 180 / Math.PI;
        bearing = (bearing + 360) % 360;

        return bearing;
    }

    // 无人机移动动画
    animateDroneMovement(start, end, duration, callback) {
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // 使用线性插值计算当前位置
            const lat = start[0] + (end[0] - start[0]) * progress;
            const lng = start[1] + (end[1] - start[1]) * progress;

            this.droneMarker.setLatLng([lat, lng]);

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else if (callback) {
                callback();
            }
        };

        requestAnimationFrame(animate);
    }

    // 初始化事件监听器
    initializeEventListeners() {
        try {
            if (this.eventListenersInitialized) {
                return; // 如果已经初始化过，直接返回
            }

            // 任务控制按钮
            const pauseBtn = document.getElementById('pauseTask');
            const resumeBtn = document.getElementById('resumeTask');
            const cancelBtn = document.getElementById('cancelTask');
            const scheduleBtn = document.getElementById('scheduleTask');
            const reportBtn = document.getElementById('generateReport');
            const annotationBtn = document.querySelector('#dataAnnotationModal .btn-primary');

            if (pauseBtn) pauseBtn.addEventListener('click', () => this.pauseTask());
            if (resumeBtn) resumeBtn.addEventListener('click', () => this.resumeTask());
            if (cancelBtn) cancelBtn.addEventListener('click', () => this.cancelTask());
            if (scheduleBtn) scheduleBtn.addEventListener('click', () => this.scheduleNewTask());
            if (reportBtn) reportBtn.addEventListener('click', () => this.generateInspectionReport());
            if (annotationBtn) annotationBtn.addEventListener('click', () => this.saveAnnotation());

            // 数据视图切换按钮
            const viewButtons = document.querySelectorAll('[data-view]');
            viewButtons.forEach(button => {
                button.addEventListener('click', () => this.switchDataView(button.dataset.view));
            });

            // 初始化任务类型监听
            const taskTypeSelect = document.getElementById('taskType');
            if (taskTypeSelect) {
                taskTypeSelect.addEventListener('change', () => {
                    const selectedTask = taskTypeSelect.value;
                    // 根据任务类型自动切换数据视图
                    if (selectedTask === 'regular' || selectedTask === 'emergency') {
                        this.switchDataView('device-data');
                    } else if (selectedTask === 'weather') {
                        this.switchDataView('weather-data');
                    }
                });
            }
            
            // 初始化无人机控制界面
            this.initializeDroneControls();

            // 巡检报告相关按钮
            const previewReportBtn = document.getElementById('previewReport');
            const generateReportBtn = document.getElementById('generateReport');
            const printReportBtn = document.getElementById('printReport');

            if (previewReportBtn) {
                previewReportBtn.addEventListener('click', () => this.previewInspectionReport());
            }
            if (generateReportBtn) {
                generateReportBtn.addEventListener('click', () => this.generateInspectionReport());
            }
            if (printReportBtn) {
                printReportBtn.addEventListener('click', () => this.printReport());
            }

            this.eventListenersInitialized = true;
        } catch (error) {
            console.error('Error initializing event listeners:', error);
        }
    }
    
    // 初始化无人机控制界面
    initializeDroneControls() {
        if (this.droneControlsInitialized) {
            return; // 如果已经初始化过，直接返回
        }

        // 定义控制按钮配置
        const controlButtons = {
            'takeoff': { id: 'takeoff', handler: () => this.takeoff() },
            'land': { id: 'land', handler: () => this.land() },
            'returnHome': { id: 'returnHome', handler: () => this.returnHome() },
            'manualMode': { id: 'manualMode', handler: () => this.switchFlightMode('manual') },
            'autoMode': { id: 'autoMode', handler: () => this.switchFlightMode('auto') },
            'followMode': { id: 'followMode', handler: () => this.switchFlightMode('follow') },
            'switchCamera': { id: 'switchCamera', handler: () => this.switchCamera() },
            'takePhoto': { id: 'takePhoto', handler: () => this.takePhoto() },
            'startRecord': { id: 'startRecord', handler: () => this.startRecording() },
            'stopRecord': { id: 'stopRecord', handler: () => this.stopRecording() },
            'emergencyStop': { id: 'emergencyStop', handler: () => this.emergencyStop() },
            'emergencyLand': { id: 'emergencyLand', handler: () => this.emergencyLand() }
        };

        // 移除所有现有的事件监听器并重新绑定
        Object.entries(controlButtons).forEach(([key, config]) => {
            const button = document.getElementById(config.id);
            if (button) {
                // 移除现有的事件监听器
                const newButton = button.cloneNode(true);
                button.parentNode.replaceChild(newButton, button);
                
                // 添加新的事件监听器
                newButton.addEventListener('click', config.handler);
            }
        });

        // 处理输入控件
        const inputs = {
            'flightHeight': { id: 'flightHeight', handler: (e) => this.setFlightHeight(e.target.value) },
            'flightSpeed': { id: 'flightSpeed', handler: (e) => this.setFlightSpeed(e.target.value) }
        };

        Object.entries(inputs).forEach(([key, config]) => {
            const input = document.getElementById(config.id);
            if (input) {
                // 移除现有的事件监听器
                const newInput = input.cloneNode(true);
                input.parentNode.replaceChild(newInput, input);
                
                // 添加新的事件监听器
                newInput.addEventListener('change', config.handler);
            }
        });
        
        // 初始化无人机状态
        this.droneFlightStatus = 'standby';
        this.droneGpsStatus = 'locked';
        this.flightMode = 'manual';
        this.currentCamera = 'optical';
        this.isRecording = false;
        
        // 更新状态显示
        this.updateDroneStatusDisplay();
        
        // 标记为已初始化
        this.droneControlsInitialized = true;
    }
    
    // 更新无人机状态显示
    updateDroneStatusDisplay() {
        const flightStatusEl = document.getElementById('flightStatus');
        const gpsStatusEl = document.getElementById('gpsStatus');
        
        if (flightStatusEl) {
            let statusText = '';
            let statusClass = '';
            
            switch (this.droneFlightStatus) {
                case 'standby':
                    statusText = '待机';
                    statusClass = 'bg-secondary';
                    break;
                case 'takeoff':
                    statusText = '起飞中';
                    statusClass = 'bg-primary';
                    break;
                case 'flying':
                    statusText = '飞行中';
                    statusClass = 'bg-success';
                    break;
                case 'landing':
                    statusText = '降落中';
                    statusClass = 'bg-warning';
                    break;
                case 'returning':
                    statusText = '返航中';
                    statusClass = 'bg-info';
                    break;
                case 'emergency':
                    statusText = '紧急状态';
                    statusClass = 'bg-danger';
                    break;
                default:
                    statusText = '未知';
                    statusClass = 'bg-secondary';
            }
            
            flightStatusEl.textContent = statusText;
            flightStatusEl.className = `badge ${statusClass}`;
        }
        
        if (gpsStatusEl) {
            let gpsText = '';
            let gpsClass = '';
            
            switch (this.droneGpsStatus) {
                case 'locked':
                    gpsText = '已定位';
                    gpsClass = 'bg-success';
                    break;
                case 'seeking':
                    gpsText = '搜索中';
                    gpsClass = 'bg-warning';
                    break;
                case 'lost':
                    gpsText = '信号丢失';
                    gpsClass = 'bg-danger';
                    break;
                default:
                    gpsText = '未知';
                    gpsClass = 'bg-secondary';
            }
            
            gpsStatusEl.textContent = gpsText;
            gpsStatusEl.className = `badge ${gpsClass}`;
        }
        
        // 更新飞行模式按钮的激活状态
        this.updateModeButtonsState();
    }
    
    // 更新模式按钮状态
    updateModeButtonsState() {
        const modeButtons = {
            'manual': document.getElementById('manualMode'),
            'auto': document.getElementById('autoMode'),
            'follow': document.getElementById('followMode')
        };
        
        // 清除所有按钮的激活状态
        Object.values(modeButtons).forEach(button => {
            if (button) {
                button.classList.remove('active');
                button.classList.remove('btn-secondary');
                button.classList.add('btn-outline-secondary');
            }
        });
        
        // 设置当前模式按钮的激活状态
        const activeButton = modeButtons[this.flightMode];
        if (activeButton) {
            activeButton.classList.add('active');
            activeButton.classList.remove('btn-outline-secondary');
            activeButton.classList.add('btn-secondary');
        }
    }
    
    // 起飞功能
    takeoff() {
        if (this.droneFlightStatus !== 'flying' && this.droneFlightStatus !== 'takeoff') {
            this.droneFlightStatus = 'takeoff';
            this.updateDroneStatusDisplay();
            
            // 显示起飞操作反馈
            this.showAlert('无人机正在起飞...', 'info');
            
            // 模拟起飞过程
            setTimeout(() => {
                this.droneFlightStatus = 'flying';
                this.updateDroneStatusDisplay();
                this.showAlert('无人机已成功起飞', 'success');
                
                // 更新地图上的无人机位置（略微上升）
                if (this.droneMarker) {
                    const currentLatLng = this.droneMarker.getLatLng();
                    this.droneMarker.setLatLng([
                        currentLatLng.lat + 0.0001, 
                        currentLatLng.lng + 0.0001
                    ]);
                }
                
                // 更新高度数据
                this.baseValues.droneHeight = parseFloat(document.getElementById('flightHeight').value);
            }, 2000);
        } else {
            this.showAlert('无人机已在飞行状态', 'warning');
        }
    }
    
    // 降落功能
    land() {
        if (this.droneFlightStatus === 'flying' || this.droneFlightStatus === 'returning') {
            this.droneFlightStatus = 'landing';
            this.updateDroneStatusDisplay();
            
            // 显示降落操作反馈
            this.showAlert('无人机正在降落...', 'info');
            
            // 模拟降落过程
            setTimeout(() => {
                this.droneFlightStatus = 'standby';
                this.updateDroneStatusDisplay();
                this.showAlert('无人机已成功降落', 'success');
                
                // 更新高度数据
                this.baseValues.droneHeight = 0;
            }, 2000);
        } else {
            this.showAlert('无人机不在飞行状态', 'warning');
        }
    }
    
    // 返航功能
    returnHome() {
        if (this.droneFlightStatus === 'flying') {
            this.droneFlightStatus = 'returning';
            this.updateDroneStatusDisplay();
            
            // 显示返航操作反馈
            this.showAlert('无人机正在返航...', 'info');
            
            // 模拟返航过程
            setTimeout(() => {
                // 返回初始位置
                if (this.droneMarker && this.map) {
                    const homePosition = this.map.getCenter();
                    this.animateDroneMovement(
                        [this.droneMarker.getLatLng().lat, this.droneMarker.getLatLng().lng],
                        [homePosition.lat, homePosition.lng],
                        3000,
                        () => {
                            this.droneFlightStatus = 'landing';
                            this.updateDroneStatusDisplay();
                            
                            setTimeout(() => {
                                this.droneFlightStatus = 'standby';
                                this.updateDroneStatusDisplay();
                                this.showAlert('无人机已返航并降落', 'success');
                                
                                // 更新高度数据
                                this.baseValues.droneHeight = 0;
                            }, 1500);
                        }
                    );
                } else {
                    this.droneFlightStatus = 'standby';
                    this.updateDroneStatusDisplay();
                    this.showAlert('无人机已返航并降落', 'success');
                }
            }, 1000);
        } else {
            this.showAlert('无人机不在飞行状态', 'warning');
        }
    }
    
    // 切换飞行模式
    switchFlightMode(mode) {
        // 只有在飞行状态才能切换模式
        if (this.droneFlightStatus === 'flying' || this.droneFlightStatus === 'standby') {
            this.flightMode = mode;
            this.updateDroneStatusDisplay();
            
            // 显示模式切换反馈
            let modeName = '';
            switch (mode) {
                case 'manual':
                    modeName = '手动模式';
                    break;
                case 'auto':
                    modeName = '自动巡检';
                    // 如果是自动模式，可以启动预设巡检路径
                    if (this.droneFlightStatus === 'flying') {
                        this.startInspectionRoute();
                    }
                    break;
                case 'follow':
                    modeName = '跟随模式';
                    break;
            }
            
            this.showAlert(`已切换到${modeName}`, 'info');
        } else {
            this.showAlert('只有在待机或飞行状态才能切换模式', 'warning');
        }
    }
    
    // 启动巡检路径
    startInspectionRoute() {
        if (this.flightMode === 'auto' && this.droneFlightStatus === 'flying') {
            this.showAlert('正在按预设路径开始巡检', 'info');
            
            // 重新开始无人机路径动画
            this.startDroneAnimation();
        }
    }
    
    // 切换相机
    switchCamera() {
        if (this.currentCamera === 'optical') {
            this.currentCamera = 'thermal';
            this.showAlert('已切换到热成像相机', 'info');
            
            // 更新视频显示
            const videoStreams = document.querySelectorAll('.nav-link');
            videoStreams.forEach(stream => {
                if (stream.textContent.includes('热成像')) {
                    stream.click();
                }
            });
        } else {
            this.currentCamera = 'optical';
            this.showAlert('已切换到可见光相机', 'info');
            
            // 更新视频显示
            const videoStreams = document.querySelectorAll('.nav-link');
            videoStreams.forEach(stream => {
                if (stream.textContent.includes('视频')) {
                    stream.click();
                }
            });
        }
    }
    
    // 拍照功能
    takePhoto() {
        if (this.droneFlightStatus === 'flying') {
            const cameraType = this.currentCamera === 'optical' ? '可见光' : '热成像';
            this.showAlert(`${cameraType}相机拍照成功`, 'success');
            
            // 模拟拍照动画效果
            const videoElement = document.querySelector('.ratio');
            if (videoElement) {
                const flash = document.createElement('div');
                flash.style.position = 'absolute';
                flash.style.top = '0';
                flash.style.left = '0';
                flash.style.width = '100%';
                flash.style.height = '100%';
                flash.style.backgroundColor = 'white';
                flash.style.opacity = '0.7';
                flash.style.transition = 'opacity 0.5s';
                flash.style.zIndex = '100';
                
                videoElement.appendChild(flash);
                
                setTimeout(() => {
                    flash.style.opacity = '0';
                    setTimeout(() => {
                        videoElement.removeChild(flash);
                    }, 500);
                }, 100);
            }
        } else {
            this.showAlert('无人机不在飞行状态，无法拍照', 'warning');
        }
    }
    
    // 开始录像
    startRecording() {
        if (this.droneFlightStatus === 'flying' && !this.isRecording) {
            this.isRecording = true;
            
            const cameraType = this.currentCamera === 'optical' ? '可见光' : '热成像';
            this.showAlert(`${cameraType}相机开始录像`, 'info');
            
            // 更改录像按钮状态
            const startRecordBtn = document.getElementById('startRecord');
            const stopRecordBtn = document.getElementById('stopRecord');
            
            if (startRecordBtn) {
                startRecordBtn.classList.remove('btn-outline-info');
                startRecordBtn.classList.add('btn-info');
            }
            
            // 开始显示录像指示器
            this.startRecordingIndicator();
        } else if (this.isRecording) {
            this.showAlert('已经在录像中', 'warning');
        } else {
            this.showAlert('无人机不在飞行状态，无法录像', 'warning');
        }
    }
    
    // 显示录像指示器
    startRecordingIndicator() {
        const videoElement = document.querySelector('.ratio');
        if (videoElement && !document.querySelector('.recording-indicator')) {
            const indicator = document.createElement('div');
            indicator.className = 'recording-indicator';
            indicator.style.position = 'absolute';
            indicator.style.top = '10px';
            indicator.style.left = '10px';
            indicator.style.width = '15px';
            indicator.style.height = '15px';
            indicator.style.backgroundColor = 'red';
            indicator.style.borderRadius = '50%';
            indicator.style.animation = 'blink 1s infinite';
            
            // 添加动画样式
            const style = document.createElement('style');
            style.textContent = `
                @keyframes blink {
                    0% { opacity: 1; }
                    50% { opacity: 0.3; }
                    100% { opacity: 1; }
                }
            `;
            document.head.appendChild(style);
            
            // 添加录像计时器
            const timer = document.createElement('div');
            timer.className = 'recording-timer';
            timer.style.position = 'absolute';
            timer.style.top = '10px';
            timer.style.left = '30px';
            timer.style.color = 'red';
            timer.style.fontWeight = 'bold';
            
            let seconds = 0;
            this.recordingTimer = setInterval(() => {
                seconds++;
                const minutes = Math.floor(seconds / 60);
                const remainingSeconds = seconds % 60;
                timer.textContent = `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
            }, 1000);
            
            videoElement.appendChild(indicator);
            videoElement.appendChild(timer);
        }
    }
    
    // 停止录像
    stopRecording() {
        if (this.isRecording) {
            this.isRecording = false;
            
            // 清除录像计时器
            if (this.recordingTimer) {
                clearInterval(this.recordingTimer);
                this.recordingTimer = null;
            }
            
            // 移除录像指示器
            const indicator = document.querySelector('.recording-indicator');
            const timer = document.querySelector('.recording-timer');
            if (indicator) indicator.remove();
            if (timer) timer.remove();
            
            // 更改录像按钮状态
            const startRecordBtn = document.getElementById('startRecord');
            if (startRecordBtn) {
                startRecordBtn.classList.remove('btn-info');
                startRecordBtn.classList.add('btn-outline-info');
            }
            
            const cameraType = this.currentCamera === 'optical' ? '可见光' : '热成像';
            this.showAlert(`${cameraType}相机停止录像，录像已保存`, 'success');
        } else {
            this.showAlert('当前没有进行录像', 'warning');
        }
    }
    
    // 设置飞行高度
    setFlightHeight(height) {
        height = parseFloat(height);
        
        // 限制合理范围
        if (height < 0) height = 0;
        if (height > 500) height = 500;
        
        // 更新显示
        document.getElementById('flightHeight').value = height;
        
        if (this.droneFlightStatus === 'flying') {
            this.showAlert(`飞行高度已设置为 ${height}m`, 'info');
            
            // 逐渐调整到目标高度
            const currentHeight = this.baseValues.droneHeight;
            const duration = Math.abs(height - currentHeight) * 100; // 高度差越大，调整时间越长
            
            this.animateHeightChange(currentHeight, height, duration);
        } else {
            // 仅设置预设高度，不实际执行
            this.baseValues.droneHeight = height;
        }
    }
    
    // 高度变化动画
    animateHeightChange(start, end, duration) {
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // 使用线性插值计算当前高度
            const currentHeight = start + (end - start) * progress;
            this.baseValues.droneHeight = currentHeight;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                this.baseValues.droneHeight = end;
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    // 设置飞行速度
    setFlightSpeed(speed) {
        speed = parseFloat(speed);
        
        // 限制合理范围
        if (speed < 0) speed = 0;
        if (speed > 20) speed = 20;
        
        // 更新显示
        document.getElementById('flightSpeed').value = speed;
        
        // 更新无人机速度
        this.baseValues.droneSpeed = speed;
        
        if (this.droneFlightStatus === 'flying') {
            this.showAlert(`飞行速度已设置为 ${speed}m/s`, 'info');
        }
    }
    
    // 紧急停止
    emergencyStop() {
        // 使用自定义模态框替代原生confirm
        const emergencyStopModal = new bootstrap.Modal(document.getElementById('emergencyStopModal'));
        emergencyStopModal.show();
        
        // 添加确认按钮的事件监听器
        const confirmBtn = document.getElementById('confirmEmergencyStop');
        
        // 移除之前可能存在的事件监听器
        const newConfirmBtn = confirmBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
        
        // 添加新的事件监听器
        newConfirmBtn.addEventListener('click', () => {
            // 保存当前状态用于恢复
            const previousStatus = this.droneFlightStatus;
            
            // 设置为紧急状态
            this.droneFlightStatus = 'emergency';
            this.updateDroneStatusDisplay();
            
            // 显示警告
            this.showAlert('执行紧急停止！所有操作已中断', 'danger');
            
            // 停止所有动画和操作
            if (this.isRecording) {
                this.stopRecording();
            }
            
            // 需要手动重置到待机状态
            setTimeout(() => {
                this.showAlert('无人机已安全停止，需要手动重置', 'warning');
            }, 3000);
            
            // 关闭模态框
            emergencyStopModal.hide();
        });
    }
    
    // 紧急降落
    emergencyLand() {
        if (this.droneFlightStatus === 'flying' || this.droneFlightStatus === 'returning' || this.droneFlightStatus === 'takeoff') {
            this.droneFlightStatus = 'landing';
            this.updateDroneStatusDisplay();
            
            // 显示警告
            this.showAlert('执行紧急降落！无人机将在当前位置降落', 'warning');
            
            // 快速降落动画
            setTimeout(() => {
                this.droneFlightStatus = 'standby';
                this.updateDroneStatusDisplay();
                this.showAlert('无人机已紧急降落', 'success');
                
                // 更新高度数据
                this.baseValues.droneHeight = 0;
            }, 1000);
        } else {
            this.showAlert('无人机不在飞行状态', 'warning');
        }
    }

    // 切换数据视图
    switchDataView(viewType) {
        const deviceSection = document.getElementById('device-monitoring');
        const weatherSection = document.getElementById('weather-monitoring');
        const deviceButton = document.querySelector('[data-view="device-data"]');
        const weatherButton = document.querySelector('[data-view="weather-data"]');
        const droneStatus = document.getElementById('drone-status');

        // 确保无人机状态始终可见
        if (droneStatus) {
            droneStatus.style.display = 'block';
        }

        if (viewType === 'device-data') {
            deviceSection.style.display = 'block';
            weatherSection.style.display = 'none';
            deviceButton.classList.add('active');
            weatherButton.classList.remove('active');
            this.currentMode = 'inspection';
        } else if (viewType === 'weather-data') {
            deviceSection.style.display = 'none';
            weatherSection.style.display = 'block';
            deviceButton.classList.remove('active');
            weatherButton.classList.add('active');
            this.currentMode = 'weather';
        }

        // 不再调用 updateDataPollingRate，因为数据更新应该持续进行
        // 只需要更新当前模式即可
    }

    // 更新数据轮询频率
    updateDataPollingRate() {
        // 清除现有的更新间隔
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        // 创建新的更新间隔，同时更新所有数据
        this.updateInterval = setInterval(() => {
            try {
                // 始终更新无人机状态数据
                this.updateDroneStatus();

                // 根据当前模式更新相应的数据显示
                // 注意：我们仍然更新所有数据，只是根据当前模式决定显示哪些数据
                this.updateDeviceData();
                this.updateWeatherData();

                // 如果需要，可以在这里添加其他数据更新逻辑
            } catch (error) {
                console.error('Error in data update interval:', error);
            }
        }, 1000); // 统一使用1秒的更新频率
    }

    // 更新无人机状态数据
    updateDroneStatus() {
        try {
            // 更新无人机状态
            const height = this.updateValue(this.baseValues.droneHeight, this.fluctuationRanges.droneHeight);
            const speed = this.updateValue(this.baseValues.droneSpeed, this.fluctuationRanges.droneSpeed);
            const battery = Math.min(100, this.updateValue(this.baseValues.batteryLevel, this.fluctuationRanges.batteryLevel));
            const signal = this.updateValue(this.baseValues.signalStrength, this.fluctuationRanges.signalStrength, true);

            // 更新显示
            const heightEl = document.querySelector('[data-drone="height"] .sensor-value');
            const speedEl = document.querySelector('[data-drone="speed"] .sensor-value');
            const batteryEl = document.querySelector('[data-drone="battery"] .sensor-value');
            const batteryBarEl = document.querySelector('[data-drone="battery"] .progress-bar');
            const signalEl = document.querySelector('[data-drone="signal"] .sensor-value');

            if (heightEl) heightEl.textContent = `${height.toFixed(1)}m`;
            if (speedEl) speedEl.textContent = `${speed.toFixed(1)}m/s`;
            if (batteryEl) batteryEl.textContent = `${Math.round(battery)}%`;
            if (batteryBarEl) {
                batteryBarEl.style.width = `${battery}%`;
                batteryBarEl.className = `progress-bar ${battery < 20 ? 'bg-danger' : battery < 40 ? 'bg-warning' : 'bg-success'}`;
            }
            if (signalEl) signalEl.textContent = `${Math.round(signal)}dBm`;

            // 更新信号强度图标
            const signalIcon = document.querySelector('[data-drone="signal"] .signal-strength i');
            if (signalIcon) {
                const signalClass = signal >= -60 ? 'reception-4' : 
                                  signal >= -70 ? 'reception-3' : 
                                  signal >= -80 ? 'reception-2' : 
                                  signal >= -90 ? 'reception-1' : 'reception-0';
                signalIcon.className = `bi bi-${signalClass} ${signal >= -80 ? 'text-success' : signal >= -90 ? 'text-warning' : 'text-danger'}`;
            }
        } catch (error) {
            console.error('Error updating drone status:', error);
        }
    }

    // 更新设备数据
    updateDeviceData() {
        try {
            // 更新所有传感器数据
            const sensors = {
                'surface-temp': { value: this.updateValue(this.baseValues.surfaceTemp, this.fluctuationRanges.surfaceTemp), unit: '°C' },
                'temp-diff': { value: this.updateValue(this.baseValues.tempDiff, this.fluctuationRanges.tempDiff, true), unit: '°C' },
                'hot-spot': { value: Math.max(0, Math.round(this.updateValue(this.baseValues.hotSpotCount, this.fluctuationRanges.hotSpotCount, true))), unit: '' },
                'surface-damage': { value: this.updateValue(this.baseValues.surfaceDamage, this.fluctuationRanges.surfaceDamage), unit: '%' },
                'contamination': { value: this.updateValue(this.baseValues.contamination, this.fluctuationRanges.contamination), unit: '%' },
                'deformation': { value: this.updateValue(this.baseValues.deformation, this.fluctuationRanges.deformation), unit: 'mm' },
                'vibration': { value: this.updateValue(this.baseValues.vibration, this.fluctuationRanges.vibration), unit: 'g' },
                'noise': { value: this.updateValue(this.baseValues.noise, this.fluctuationRanges.noise), unit: 'dB' },
                'resonance': { value: this.updateValue(this.baseValues.resonance, this.fluctuationRanges.resonance), unit: 'Hz' },
                'corona': { value: this.updateValue(this.baseValues.corona, this.fluctuationRanges.corona), unit: 'dB' },
                'blade-angle': { value: this.updateValue(this.baseValues.bladeAngle, this.fluctuationRanges.bladeAngle, true), unit: '°' },
                'panel-tilt': { value: this.updateValue(this.baseValues.panelTilt, this.fluctuationRanges.panelTilt), unit: '°' }
            };

            // 更新显示
            Object.entries(sensors).forEach(([id, data]) => {
                const el = document.querySelector(`[data-sensor="${id}"] .sensor-value`);
                if (el) {
                    el.textContent = `${Number(data.value).toFixed(2)}${data.unit}`;
                }

                // 更新状态标签
                const badge = document.querySelector(`[data-sensor="${id}"] .status-badge`);
                if (badge && this.thresholds[id.replace('-', '')]) {
                    const threshold = this.thresholds[id.replace('-', '')];
                    let className, text;
                    if (data.value >= threshold.high) {
                        className = 'bg-warning';
                        text = '偏高';
                    } else if (data.value <= threshold.low) {
                        className = 'bg-danger';
                        text = '偏低';
                    } else {
                        className = 'bg-success';
                        text = '正常';
                    }
                    badge.className = `badge ${className} status-badge`;
                    badge.textContent = text;
                }
            });

            // 特殊处理绝缘状态
            const insulationEl = document.querySelector('[data-sensor="insulation"] .sensor-value');
            const insulationBadge = document.querySelector('[data-sensor="insulation"] .status-badge');
            if (insulationEl && insulationBadge) {
                const isNormal = this.baseValues.insulation === 1;
                insulationEl.textContent = isNormal ? '正常' : '异常';
                insulationBadge.className = `badge ${isNormal ? 'bg-success' : 'bg-danger'} status-badge`;
                insulationBadge.textContent = isNormal ? '正常' : '异常';
            }
        } catch (error) {
            console.error('Error updating device data:', error);
        }
    }

    // 更新气象数据
    updateWeatherData() {
        // 原有的气象数据更新代码
        const weatherSensors = {
            'temperature': { value: this.updateValue(this.baseValues.envTemp, this.fluctuationRanges.envTemp), unit: '°C' },
            // ... 其他气象传感器数据 ...
        };

        // 更新显示
        Object.entries(weatherSensors).forEach(([id, data]) => {
            const el = document.querySelector(`[data-weather="${id}"] .sensor-value`);
            if (el) {
                el.textContent = `${Number(data.value).toFixed(2)}${data.unit}`;
            }
        });

        // 更新时序数据
        const timeSeriesData = {
            timestamp: new Date().toISOString(),
            location: {
                latitude: this.droneMarker?.getLatLng().lat,
                longitude: this.droneMarker?.getLatLng().lng,
                altitude: this.baseValues.droneHeight
            },
            weather: Object.fromEntries(
                Object.entries(weatherSensors).map(([key, data]) => [key, data.value])
            ),
            energy_related: {
                solar_potential: weatherSensors['solar-radiation'].value * 
                    (1 - weatherSensors['cloud-cover'].value / 100) * 
                    weatherSensors['clearness-index'].value,
                wind_potential: 0.5 * weatherSensors['air-density'].value * 
                    Math.pow(weatherSensors['wind-speed'].value, 3)
            }
        };

        // 可以在这里添加数据导出逻辑
        if (this.currentMode === 'weather') {
            console.log('Weather time series data:', timeSeriesData);
        }
    }

    // 初始化WebSocket连接
    initializeWebSocket() {
        try {
            this.ws = new WebSocket('ws://your-websocket-server');
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    switch(data.type) {
                        case 'dronePosition':
                            this.updateDronePosition(data.position);
                            break;
                        case 'sensorData':
                            this.updateSensorData(data.data);
                            break;
                        case 'videoStream':
                            this.updateVideoStream(data.url);
                            break;
                        case 'thermalData':
                            this.updateThermalView(data.data);
                            break;
                        case 'alert':
                            this.showAlert(data.message);
                            break;
                    }
                } catch (error) {
                    console.error('Error processing WebSocket message:', error);
                }
            };
        } catch (error) {
            console.warn('WebSocket initialization failed:', error);
        }
    }

    // 开始数据轮询
    startDataPolling() {
        setInterval(() => {
            this.updateTaskProgress();
            this.updateBatteryStatus();
            this.updateWeatherInfo();
        }, 5000);
    }

    // 更新无人机位置
    updateDronePosition(position) {
        const { lat, lng, altitude, heading } = position;
        this.droneMarker.setLatLng([lat, lng]);
        // 更新无人机朝向
        this.droneMarker.setRotationAngle(heading);
    }

    // 更新传感器数据
    updateSensorData(data) {
        const { temperature, vibration, noise } = data;
        document.querySelector('[data-sensor="temperature"] .value').textContent = `${temperature}°C`;
        document.querySelector('[data-sensor="vibration"] .value').textContent = `${vibration}g`;
        document.querySelector('[data-sensor="noise"] .value').textContent = `${noise}dB`;
    }

    // 更新视频流
    updateVideoStream(url) {
        const videoElement = document.getElementById('videoStream');
        if (videoElement) {
            videoElement.innerHTML = `<video src="${url}" autoplay></video>`;
        }
    }

    // 更新热成像视图
    updateThermalView(data) {
        // 使用热成像数据更新Canvas
        const canvas = document.getElementById('thermalCanvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            // 绘制热成像图
            this.drawThermalImage(ctx, data);
        }
    }

    // 暂停任务
    pauseTask() {
        this.isPaused = true;
        this.ws.send(JSON.stringify({
            type: 'command',
            action: 'pause'
        }));
        this.updateTaskStatus('已暂停');
    }

    // 继续任务
    resumeTask() {
        this.isPaused = false;
        this.ws.send(JSON.stringify({
            type: 'command',
            action: 'resume'
        }));
        this.updateTaskStatus('进行中');
    }

    // 取消任务
    cancelTask() {
        // 显示取消确认模态框
        const cancelTaskModal = new bootstrap.Modal(document.getElementById('cancelTaskModal'));
        cancelTaskModal.show();

        // 绑定确认按钮事件
        const confirmBtn = document.getElementById('confirmCancelTask');
        
        // 移除之前可能存在的事件监听器
        const newConfirmBtn = confirmBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
        
        // 添加新的事件监听器
        newConfirmBtn.addEventListener('click', () => {
            // 保存当前状态用于恢复
            const previousStatus = this.currentTask?.status;
            
            // 更新任务状态
            if (this.currentTask) {
                this.currentTask.status = 'cancelled';
            }
            
            // 更新UI显示
            this.updateTaskStatus('已取消');
            
            // 显示取消成功提示
            this.showAlert('任务已取消', 'danger');
            
            // 关闭模态框
            cancelTaskModal.hide();
            
            // 重置进度条
            const progressBar = document.querySelector('.progress-bar');
            if (progressBar) {
                progressBar.style.width = '0%';
                progressBar.textContent = '0%';
            }
            
            // 如果需要，可以在这里添加其他清理工作
        });
    }

    // 更新任务状态显示
    updateTaskStatus(status) {
        // 更新状态显示
        const statusElement = document.querySelector('.card-title.mb-0');
        if (statusElement) {
            statusElement.textContent = `当前任务状态：${status}`;
        }

        // 根据状态更新按钮状态
        const pauseBtn = document.getElementById('pauseTask');
        const resumeBtn = document.getElementById('resumeTask');
        const cancelBtn = document.getElementById('cancelTask');

        if (status === '已取消') {
            if (pauseBtn) pauseBtn.disabled = true;
            if (resumeBtn) resumeBtn.disabled = true;
            if (cancelBtn) cancelBtn.disabled = true;
        }
    }

    // 安排新任务
    scheduleNewTask() {
        const taskType = document.getElementById('taskType').value;
        const priority = document.getElementById('priority').value;
        const startTime = document.getElementById('startTime').value;

        const task = {
            type: taskType,
            priority: priority,
            startTime: startTime,
            devices: this.getSelectedDevices()
        };

        this.ws.send(JSON.stringify({
            type: 'newTask',
            task: task
        }));

        this.showAlert('任务已安排', 'success');
    }

    // 获取选中的设备
    getSelectedDevices() {
        const checkboxes = document.querySelectorAll('.device-item input[type="checkbox"]:checked');
        return Array.from(checkboxes).map(cb => cb.closest('.device-item').dataset.deviceId);
    }

    // 生成巡检报告
    generateInspectionReport() {
        try {
            // 显示加载提示
            const loadingModal = new bootstrap.Modal(document.getElementById('generateReportLoadingModal'));
            loadingModal.show();
            
            // 获取进度条和状态文本元素
            const progressBar = document.getElementById('generateReportProgress');
            const statusText = document.getElementById('generateReportStatus');
            
            // 模拟生成过程
            let progress = 0;
            const updateProgress = () => {
                progress += 5;
                if (progressBar) progressBar.style.width = `${progress}%`;
                
                if (progress <= 30) {
                    if (statusText) statusText.textContent = '正在收集数据...';
                } else if (progress <= 60) {
                    if (statusText) statusText.textContent = '正在生成报告内容...';
                } else if (progress <= 90) {
                    if (statusText) statusText.textContent = '正在处理图片...';
                } else {
                    if (statusText) statusText.textContent = '正在完成报告生成...';
                }
                
                if (progress < 100) {
                    setTimeout(updateProgress, 100);
                } else {
                    // 生成完成后的操作
                    setTimeout(() => {
                        // 隐藏加载提示
                        loadingModal.hide();
                        
                        // 创建报告数据
                        const reportData = this.collectReportData();
                        
                        // 生成PDF文件名
                        const fileName = `巡检报告_${reportData.taskId}.pdf`;
                        
                        // 显示成功提示
                        const toast = new bootstrap.Toast(document.getElementById('reportGeneratedToast'));
                        toast.show();
                        
                        // 触发下载（这里仅作为示例，实际项目中需要实现真实的PDF生成）
                        this.downloadReport(fileName, reportData);
                        
                        // 重置进度条
                        if (progressBar) progressBar.style.width = '0%';
                    }, 500);
                }
            };
            
            // 开始更新进度
            updateProgress();
        } catch (error) {
            console.error('Error generating report:', error);
            this.showAlert('生成报告时发生错误', 'danger');
        }
    }

    // 打印报告
    printReport() {
        try {
            const reportData = this.collectReportData();
            const printWindow = window.open('', '_blank');
            if (!printWindow) {
                throw new Error('Unable to open print window');
            }

            printWindow.document.write(`
                <html>
                    <head>
                        <title>巡检报告 - ${reportData.taskId}</title>
                        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                        <style>
                            @media print {
                                body { padding: 20px; }
                                .no-print { display: none; }
                            }
                        </style>
                    </head>
                    <body>
                        ${this.generateReportHTML(reportData)}
                    </body>
                </html>
            `);
            
            printWindow.document.close();
            setTimeout(() => {
                printWindow.print();
                printWindow.close();
            }, 500);
        } catch (error) {
            console.error('Error printing report:', error);
            this.showAlert('打印报告时发生错误', 'danger');
        }
    }

    // 下载报告
    downloadReport(fileName, reportData) {
        try {
            // 这里应该实现实际的PDF生成逻辑
            // 当前仅作为示例，创建一个简单的文本文件
            const reportText = JSON.stringify(reportData, null, 2);
            const blob = new Blob([reportText], { type: 'application/pdf' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error downloading report:', error);
            this.showAlert('下载报告时发生错误', 'danger');
        }
    }

    // 收集报告数据
    collectReportData() {
        const reportData = {
            taskId: this.generateTaskId(),
            dateTime: new Date().toLocaleString(),
            area: '四川会理县光伏电站',
            weather: this.getWeatherSummary(),
            deviceStats: this.calculateDeviceStats(),
            anomalies: this.collectAnomalies(),
            images: this.collectInspectionImages(),
            route: this.getInspectionRoute(),
            notes: document.getElementById('reportNotes')?.value || '',
            recommendations: {
                urgent: document.getElementById('recommendationUrgent')?.checked || false,
                text: document.getElementById('reportRecommendations')?.value || ''
            }
        };

        // 更新报告界面显示
        this.updateReportDisplay(reportData);

        return reportData;
    }

    // 生成任务ID
    generateTaskId() {
        const date = new Date();
        return `INSPECT-${date.getFullYear()}${(date.getMonth() + 1).toString().padStart(2, '0')}${date.getDate().toString().padStart(2, '0')}-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`;
    }

    // 获取天气概况
    getWeatherSummary() {
        return `温度: ${this.baseValues.envTemp}°C, 湿度: ${this.baseValues.humidity}%, 风速: ${this.baseValues.windSpeed}m/s`;
    }

    // 计算设备统计数据
    calculateDeviceStats() {
        // 模拟数据，实际应从设备监测数据中统计
        return {
            normal: 42,
            warning: 3,
            error: 1,
            total: 46
        };
    }

    // 收集异常数据
    collectAnomalies() {
        // 模拟异常数据，实际应从设备监测数据中收集
        return [
            {
                deviceId: 'PANEL-A-123',
                type: '光伏板',
                anomalyType: '温度异常',
                value: '75°C',
                status: '警告',
                recommendation: '建议检查散热情况'
            },
            {
                deviceId: 'INVERTER-B-456',
                type: '逆变器',
                anomalyType: '效率降低',
                value: '85%',
                status: '异常',
                recommendation: '建议进行维护保养'
            }
        ];
    }

    // 收集巡检图片
    collectInspectionImages() {
        // 模拟图片数据，实际应从拍照记录中收集
        return [
            {
                url: 'static/image/solar_panels_it_image.png',
                description: '光伏板热成像图1',
                timestamp: new Date().toLocaleString()
            },
            {
                url: 'static/image/solar_panels_it_image.png',
                description: '光伏板可见光图1',
                timestamp: new Date().toLocaleString()
            }
        ];
    }

    // 获取巡检路线
    getInspectionRoute() {
        return this.inspectionPath || [];
    }

    // 更新报告显示
    updateReportDisplay(data) {
        // 更新基本信息
        document.getElementById('reportTaskId').value = data.taskId;
        document.getElementById('reportDateTime').value = data.dateTime;
        document.getElementById('reportArea').value = data.area;
        document.getElementById('reportWeather').value = data.weather;

        // 更新设备统计
        document.getElementById('reportNormalCount').textContent = data.deviceStats.normal;
        document.getElementById('reportWarningCount').textContent = data.deviceStats.warning;
        document.getElementById('reportErrorCount').textContent = data.deviceStats.error;
        document.getElementById('reportTotalCount').textContent = data.deviceStats.total;

        // 更新异常表格
        const tbody = document.querySelector('#reportAnomalyTable tbody');
        tbody.innerHTML = '';
        data.anomalies.forEach(anomaly => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${anomaly.deviceId}</td>
                <td>${anomaly.type}</td>
                <td>${anomaly.anomalyType}</td>
                <td>${anomaly.value}</td>
                <td><span class="badge ${this.getStatusBadgeClass(anomaly.status)}">${anomaly.status}</span></td>
                <td>${anomaly.recommendation}</td>
            `;
            tbody.appendChild(tr);
        });

        // 更新图片画廊
        const gallery = document.getElementById('reportImageGallery');
        gallery.innerHTML = '';
        data.images.forEach(image => {
            const col = document.createElement('div');
            col.className = 'col-md-4';
            col.innerHTML = `
                <div class="card">
                    <img src="${image.url}" class="card-img-top" alt="${image.description}">
                    <div class="card-body">
                        <p class="card-text small">${image.description}</p>
                        <p class="card-text"><small class="text-muted">${image.timestamp}</small></p>
                    </div>
                </div>
            `;
            gallery.appendChild(col);
        });

        // 更新巡检轨迹地图
        this.updateReportMap(data.route);
    }

    // 获取状态徽章样式
    getStatusBadgeClass(status) {
        switch (status.toLowerCase()) {
            case '正常':
                return 'bg-success';
            case '警告':
                return 'bg-warning';
            case '异常':
                return 'bg-danger';
            default:
                return 'bg-secondary';
        }
    }

    // 更新报告中的地图
    updateReportMap(route) {
        if (!this.reportMap) {
            this.reportMap = L.map('reportMap').setView([26.4296, 102.2485], 15);
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: 'Tiles &copy; Esri'
            }).addTo(this.reportMap);
        }

        // 清除现有路径
        if (this.reportPathLayer) {
            this.reportMap.removeLayer(this.reportPathLayer);
        }

        // 绘制巡检路径
        if (route && route.length > 0) {
            this.reportPathLayer = L.polyline(route, {
                color: '#FF4081',
                weight: 3,
                opacity: 0.8
            }).addTo(this.reportMap);

            // 添加起点和终点标记
            L.marker(route[0], {
                icon: L.divIcon({
                    className: 'route-marker start-marker',
                    html: '<i class="bi bi-geo-alt-fill text-success"></i>',
                    iconSize: [20, 20]
                })
            }).addTo(this.reportMap);

            L.marker(route[route.length - 1], {
                icon: L.divIcon({
                    className: 'route-marker end-marker',
                    html: '<i class="bi bi-geo-alt-fill text-danger"></i>',
                    iconSize: [20, 20]
                })
            }).addTo(this.reportMap);

            // 调整地图视图以适应路径
            this.reportMap.fitBounds(this.reportPathLayer.getBounds());
        }
    }

    // 生成PDF报告
    generatePDFReport(data) {
        // 显示加载提示
        const loadingModal = new bootstrap.Modal(document.getElementById('generateReportLoadingModal'));
        loadingModal.show();
        
        // 获取进度条和状态文本元素
        const progressBar = document.getElementById('generateReportProgress');
        const statusText = document.getElementById('generateReportStatus');
        
        // 模拟生成过程
        let progress = 0;
        const updateProgress = () => {
            progress += 5;
            progressBar.style.width = `${progress}%`;
            
            if (progress <= 30) {
                statusText.textContent = '正在收集数据...';
            } else if (progress <= 60) {
                statusText.textContent = '正在生成报告内容...';
            } else if (progress <= 90) {
                statusText.textContent = '正在处理图片...';
            } else {
                statusText.textContent = '正在完成报告生成...';
            }
            
            if (progress < 100) {
                setTimeout(updateProgress, 100);
            } else {
                // 生成完成后的操作
                setTimeout(() => {
                    // 隐藏加载提示
                    loadingModal.hide();
                    
                    // 创建并下载PDF文件
                    const blob = new Blob(['PDF report content'], { type: 'application/pdf' });
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `巡检报告_${data.taskId}.pdf`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    
                    // 显示成功提示
                    const toast = new bootstrap.Toast(document.getElementById('reportGeneratedToast'));
                    toast.show();
                    
                    // 重置进度条
                    setTimeout(() => {
                        progressBar.style.width = '0%';
                    }, 500);
                }, 500);
            }
        };
        
        // 开始更新进度
        updateProgress();
    }

    // 生成报告HTML预览
    generateReportHTML(data) {
        return `
            <div class="container">
                <div class="report-header text-center mb-5">
                    <h2 class="mb-3">无人机巡检报告</h2>
                    <p class="text-muted">${data.taskId}</p>
                </div>

                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title mb-3">基本信息</h5>
                                <table class="table table-sm">
                                    <tr>
                                        <th width="120">巡检时间：</th>
                                        <td>${data.dateTime}</td>
                                    </tr>
                                    <tr>
                                        <th>巡检区域：</th>
                                        <td>${data.area}</td>
                                    </tr>
                                    <tr>
                                        <th>天气状况：</th>
                                        <td>${data.weather}</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title mb-3">设备状态统计</h5>
                                <div class="row text-center">
                                    <div class="col-3">
                                        <div class="border-end">
                                            <h3 class="text-success">${data.deviceStats.normal}</h3>
                                            <small class="text-muted">正常设备</small>
                                        </div>
                                    </div>
                                    <div class="col-3">
                                        <div class="border-end">
                                            <h3 class="text-warning">${data.deviceStats.warning}</h3>
                                            <small class="text-muted">异常设备</small>
                                        </div>
                                    </div>
                                    <div class="col-3">
                                        <div class="border-end">
                                            <h3 class="text-danger">${data.deviceStats.error}</h3>
                                            <small class="text-muted">故障设备</small>
                                        </div>
                                    </div>
                                    <div class="col-3">
                                        <div>
                                            <h3 class="text-info">${data.deviceStats.total}</h3>
                                            <small class="text-muted">总计设备</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card mb-4">
                    <div class="card-body">
                        <h5 class="card-title mb-3">异常详情</h5>
                        <div class="table-responsive">
                            <table class="table table-bordered">
                                <thead class="table-light">
                                    <tr>
                                        <th>设备ID</th>
                                        <th>设备类型</th>
                                        <th>异常类型</th>
                                        <th>异常值</th>
                                        <th>状态</th>
                                        <th>建议</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.anomalies.map(anomaly => `
                                        <tr>
                                            <td>${anomaly.deviceId}</td>
                                            <td>${anomaly.type}</td>
                                            <td>${anomaly.anomalyType}</td>
                                            <td>${anomaly.value}</td>
                                            <td><span class="badge ${this.getStatusBadgeClass(anomaly.status)}">${anomaly.status}</span></td>
                                            <td>${anomaly.recommendation}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title mb-3">巡检图片</h5>
                                <div class="row g-2">
                                    ${data.images.map(image => `
                                        <div class="col-6">
                                            <div class="card">
                                                <img src="${image.url}" class="card-img-top" alt="${image.description}">
                                                <div class="card-body p-2">
                                                    <p class="card-text small">${image.description}</p>
                                                    <p class="card-text"><small class="text-muted">${image.timestamp}</small></p>
                                                </div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-body">
                                <h5 class="card-title mb-3">维护建议</h5>
                                ${data.recommendations.urgent ? 
                                    '<div class="alert alert-danger">建议紧急维修</div>' : ''}
                                <p>${data.recommendations.text || '无'}</p>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title mb-3">补充说明</h5>
                                <p>${data.notes || '无'}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // 保存数据标注
    saveAnnotation() {
        const annotation = {
            type: document.querySelector('#dataAnnotationModal select:first-child').value,
            severity: document.querySelector('#dataAnnotationModal select:last-child').value,
            notes: document.querySelector('#dataAnnotationModal textarea').value,
            timestamp: new Date().toISOString()
        };

        this.ws.send(JSON.stringify({
            type: 'annotation',
            data: annotation
        }));

        $('#dataAnnotationModal').modal('hide');
        this.showAlert('标注已保存', 'success');
    }

    // 更新任务进度
    updateTaskProgress() {
        if (!this.currentTask || this.isPaused) return;

        const progress = this.calculateTaskProgress();
        const progressBar = document.querySelector('.progress-bar');
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = `${progress}%`;
    }

    // 计算任务进度
    calculateTaskProgress() {
        // 根据已完成的检查点计算进度
        return Math.floor((this.currentTask?.completedPoints || 0) / (this.currentTask?.totalPoints || 1) * 100);
    }

    // 更新电池状态
    updateBatteryStatus() {
        // 模拟电池状态更新
        const batteryLevel = Math.floor(Math.random() * 20 + 80); // 80-100%
        document.querySelector('[data-status="battery"]').textContent = `${batteryLevel}%`;
    }

    // 更新天气信息
    updateWeatherInfo() {
        fetch('https://api.weather.com/your-endpoint')
            .then(response => response.json())
            .then(data => {
                // 更新天气信息显示
                this.updateWeatherDisplay(data);
            })
            .catch(error => console.error('Weather update failed:', error));
    }

    // 显示提示信息
    showAlert(message, type = 'info') {
        const notificationContainer = document.getElementById('drone-notifications');
        if (!notificationContainer) return;

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.style.marginBottom = '10px';
        alertDiv.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        alertDiv.style.animation = 'slideIn 0.5s ease-out';
        
        alertDiv.innerHTML = `
            <div class="d-flex align-items-center">
                ${this.getAlertIcon(type)}
                <div class="ms-2">${message}</div>
            </div>
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

        // 添加到容器
        notificationContainer.appendChild(alertDiv);

        // 自动消失
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => {
                alertDiv.remove();
            }, 500);
        }, 5000);
    }

    // 获取提示图标
    getAlertIcon(type) {
        switch (type) {
            case 'success':
                return '<i class="bi bi-check-circle-fill text-success"></i>';
            case 'warning':
                return '<i class="bi bi-exclamation-triangle-fill text-warning"></i>';
            case 'danger':
                return '<i class="bi bi-x-circle-fill text-danger"></i>';
            case 'info':
            default:
                return '<i class="bi bi-info-circle-fill text-info"></i>';
        }
    }

    // 添加CSS动画
    addNotificationStyles() {
        const styleSheet = document.createElement('style');
        styleSheet.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }

            #drone-notifications .alert {
                transition: opacity 0.5s ease-out;
            }

            #drone-notifications .alert.fade {
                transition: opacity 0.5s ease-out;
            }

            #drone-notifications .alert.fade:not(.show) {
                opacity: 0;
            }
        `;
        document.head.appendChild(styleSheet);
    }

    // 模拟实时数据更新
    simulateRealTimeData() {
        console.log('Starting real-time data simulation...'); // 添加调试日志
        // 基准值和波动范围
        const baseValues = {
            droneHeight: 50.5,
            droneSpeed: 5.2,
            batteryLevel: 75,
            signalStrength: -65,  // 基准信号强度为 -65dBm
            surfaceTemp: 32.5,    // 表面温度 (°C)
            tempDiff: 2.3,        // 温差异常 (°C)
            hotSpotCount: 0,      // 热点数量
            surfaceDamage: 0.5,   // 表面损伤率 (%)
            contamination: 3.2,    // 污渍覆盖率 (%)
            deformation: 0.2,     // 形变量 (mm)
            vibration: 0.15,      // 振动强度 (g)
            noise: 65,            // 噪声水平 (dB)
            resonance: 12.5,      // 共振频率 (Hz)
            corona: 15,           // 电晕放电 (dB)
            insulation: 1,        // 绝缘状态 (0-故障, 1-正常)
            bladeAngle: 15,       // 叶片角度 (°)
            panelTilt: 32,        // 光伏板倾角 (°)
            envTemp: 25.8,        // 环境温度 (°C)
            tempGradient: 0.6,    // 温度梯度 (°C/100m)
            humidity: 45,         // 相对湿度 (%)
            dewPoint: 13.2,       // 露点温度 (°C)
            windSpeed: 3.2,       // 风速 (m/s)
            windDirection: 225,    // 风向 (度)
            windShear: 0.03,      // 风切变 (s⁻¹)
            turbulence: 0.15,     // 湍流强度
            visibility: 12,       // 能见度 (km)
            pressure: 1013.25,    // 气压 (hPa)
            airDensity: 1.225,    // 空气密度 (kg/m³)
            solarRadiation: 800,   // 太阳辐射 (W/m²)
            cloudCover: 25,        // 云量 (%)
            clearnessIndex: 0.75,  // 晴空指数
            pm25: 35,             // PM2.5 (µg/m³)
            dust: 0.15            // 粉尘浓度 (mg/m³)
        };

        // 随机波动范围（百分比或绝对值）
        const fluctuationRanges = {
            droneHeight: 2,       // ±2%
            droneSpeed: 5,        // ±5%
            batteryLevel: 0.2,    // ±0.2%
            signalStrength: 5,    // ±5dBm
            surfaceTemp: 1,       // ±1%
            tempDiff: 0.5,        // ±0.5°C
            hotSpotCount: 1,      // ±1个
            surfaceDamage: 0.1,   // ±0.1%
            contamination: 0.5,    // ±0.5%
            deformation: 0.1,     // ±0.1mm
            vibration: 10,         // ±10%
            noise: 3,             // ±3dB
            resonance: 5,         // ±5%
            corona: 2,            // ±2dB
            insulation: 0,        // 不波动
            bladeAngle: 2,        // ±2°
            panelTilt: 0.5,       // ±0.5°
            envTemp: 0.5,         // ±0.5°C
            tempGradient: 0.1,    // ±0.1°C/100m
            humidity: 2,          // ±2%
            dewPoint: 0.5,         // ±0.5°C
            windSpeed: 0.5,        // ±0.5m/s
            windDirection: 10,     // ±10°
            windShear: 0.01,       // ±0.01s⁻¹
            turbulence: 0.05,      // ±0.05
            visibility: 0.5,       // ±0.5km
            pressure: 0.25,        // ±0.25hPa
            airDensity: 0.005,     // ±0.005kg/m³
            solarRadiation: 50,     // ±50W/m²
            cloudCover: 5,          // ±5%
            clearnessIndex: 0.05,   // ±0.05
            pm25: 5,               // ±5µg/m³
            dust: 0.02            // ±0.02mg/m³
        };

        // 阈值设置
        const thresholds = {
            surfaceTemp: { low: 10, high: 40 },
            tempDiff: { low: 0, high: 5 },
            hotSpotCount: { low: 0, high: 1 },
            surfaceDamage: { low: 0, high: 1 },
            contamination: { low: 0, high: 3 },
            deformation: { low: 0, high: 0.5 },
            vibration: { low: 0.05, high: 0.2 },
            noise: { low: 50, high: 75 },
            resonance: { low: 10, high: 15 },
            corona: { low: 0, high: 20 },
            bladeAngle: { low: 5, high: 25 },
            panelTilt: { low: 25, high: 40 }
        };

        // 气象数据阈值
        const weatherThresholds = {
            envTemp: { low: 10, high: 35 },
            tempGradient: { low: 0.3, high: 1.0 },
            humidity: { low: 20, high: 80 },
            dewPoint: { low: 0, high: 20 },
            windSpeed: { low: 0, high: 8 },
            windShear: { low: 0, high: 0.05 },
            turbulence: { low: 0, high: 0.2 },
            visibility: { low: 5, high: 15 },
            pressure: { low: 1000, high: 1020 },
            airDensity: { low: 1.1, high: 1.3 },
            solarRadiation: { low: 200, high: 1000 },
            cloudCover: { low: 0, high: 50 },
            clearnessIndex: { low: 0.5, high: 0.9 },
            pm25: { low: 0, high: 50 },
            dust: { low: 0, high: 0.2 }
        };

        // 更新函数 - 为信号强度添加特殊处理
        const updateValue = (baseValue, fluctuationPercentage, isSignal = false) => {
            if (isSignal) {
                // 信号强度特殊处理：直接加减dBm值而不是用百分比
                const fluctuation = (Math.random() - 0.5) * 2 * fluctuationPercentage;
                return baseValue + fluctuation;
            } else {
                // 其他值使用百分比波动
                const fluctuation = (Math.random() - 0.5) * 2 * (baseValue * fluctuationPercentage / 100);
                return Math.max(0, baseValue + fluctuation);
            }
        };

        // 更新状态函数
        const updateStatus = (value, thresholds) => {
            if (value >= thresholds.high) return ['bg-warning', '偏高'];
            if (value <= thresholds.low) return ['bg-danger', '偏低'];
            return ['bg-success', '正常'];
        };

        // 定期更新数据
        const updateInterval = setInterval(() => {
            try {
                // 更新无人机状态
                const height = updateValue(baseValues.droneHeight, fluctuationRanges.droneHeight);
                const speed = updateValue(baseValues.droneSpeed, fluctuationRanges.droneSpeed);
                const battery = Math.min(100, updateValue(baseValues.batteryLevel, fluctuationRanges.batteryLevel));
                const signal = updateValue(baseValues.signalStrength, fluctuationRanges.signalStrength, true); // 添加true参数表示这是信号强度

                // 添加调试日志
                console.log('Updating values:', { height, speed, battery, signal });

                // 更新无人机数据显示
                const heightEl = document.querySelector('[data-drone="height"] .sensor-value');
                const speedEl = document.querySelector('[data-drone="speed"] .sensor-value');
                const batteryEl = document.querySelector('[data-drone="battery"] .sensor-value');
                const batteryBarEl = document.querySelector('[data-drone="battery"] .progress-bar');
                const signalEl = document.querySelector('[data-drone="signal"] .sensor-value');

                // 添加调试日志
                console.log('Found elements:', {
                    heightEl: !!heightEl,
                    speedEl: !!speedEl,
                    batteryEl: !!batteryEl,
                    batteryBarEl: !!batteryBarEl,
                    signalEl: !!signalEl
                });

                if (heightEl) heightEl.textContent = `${height.toFixed(1)}m`;
                if (speedEl) speedEl.textContent = `${speed.toFixed(1)}m/s`;
                if (batteryEl) batteryEl.textContent = `${Math.round(battery)}%`;
                if (batteryBarEl) {
                    batteryBarEl.style.width = `${battery}%`;
                    batteryBarEl.className = `progress-bar ${battery < 20 ? 'bg-danger' : battery < 40 ? 'bg-warning' : 'bg-success'}`;
                }
                if (signalEl) signalEl.textContent = `${Math.round(signal)}dBm`;

                // 更新信号强度图标的状态
                const signalIcon = document.querySelector('[data-drone="signal"] .signal-strength i');
                if (signalIcon) {
                    // 根据信号强度更新图标样式
                    const signalClass = signal >= -60 ? 'reception-4' : 
                                      signal >= -70 ? 'reception-3' : 
                                      signal >= -80 ? 'reception-2' : 
                                      signal >= -90 ? 'reception-1' : 'reception-0';
                    signalIcon.className = `bi bi-${signalClass} ${signal >= -80 ? 'text-success' : signal >= -90 ? 'text-warning' : 'text-danger'}`;
                }

                // 更新设备监测数据
                const updateSensorValue = (sensorId, value, unit = '', isInteger = false) => {
                    const el = document.querySelector(`[data-sensor="${sensorId}"] .sensor-value`);
                    if (el) {
                        el.textContent = isInteger ? 
                            `${Math.round(value)}${unit}` : 
                            `${value.toFixed(1)}${unit}`;
                    }
                };

                // 更新状态标签
                const updateStatusBadge = (sensorId, value, thresholds) => {
                    const badge = document.querySelector(`[data-sensor="${sensorId}"] .status-badge`);
                    if (badge) {
                        let className, text;
                        if (value >= thresholds.high) {
                            className = 'bg-warning';
                            text = '偏高';
                        } else if (value <= thresholds.low) {
                            className = 'bg-danger';
                            text = '偏低';
                        } else {
                            className = 'bg-success';
                            text = '正常';
                        }
                        badge.className = `badge ${className} status-badge`;
                        badge.textContent = text;
                    }
                };

                // 更新所有传感器数据
                const sensors = {
                    'surface-temp': { value: updateValue(baseValues.surfaceTemp, fluctuationRanges.surfaceTemp), unit: '°C' },
                    'temp-diff': { value: updateValue(baseValues.tempDiff, fluctuationRanges.tempDiff, true), unit: '°C' },
                    'hot-spot': { value: Math.max(0, Math.round(updateValue(baseValues.hotSpotCount, fluctuationRanges.hotSpotCount, true))), unit: '' },
                    'surface-damage': { value: updateValue(baseValues.surfaceDamage, fluctuationRanges.surfaceDamage), unit: '%' },
                    'contamination': { value: updateValue(baseValues.contamination, fluctuationRanges.contamination), unit: '%' },
                    'deformation': { value: updateValue(baseValues.deformation, fluctuationRanges.deformation), unit: 'mm' },
                    'vibration': { value: updateValue(baseValues.vibration, fluctuationRanges.vibration), unit: 'g' },
                    'noise': { value: updateValue(baseValues.noise, fluctuationRanges.noise), unit: 'dB' },
                    'resonance': { value: updateValue(baseValues.resonance, fluctuationRanges.resonance), unit: 'Hz' },
                    'corona': { value: updateValue(baseValues.corona, fluctuationRanges.corona), unit: 'dB' },
                    'blade-angle': { value: updateValue(baseValues.bladeAngle, fluctuationRanges.bladeAngle, true), unit: '°' },
                    'panel-tilt': { value: updateValue(baseValues.panelTilt, fluctuationRanges.panelTilt), unit: '°' }
                };

                // 更新显示
                Object.entries(sensors).forEach(([id, data]) => {
                    updateSensorValue(id, data.value, data.unit);
                    if (thresholds[id.replace('-', '')]) {
                        updateStatusBadge(id, data.value, thresholds[id.replace('-', '')]);
                    }
                });

                // 特殊处理绝缘状态
                const insulationEl = document.querySelector('[data-sensor="insulation"] .sensor-value');
                const insulationBadge = document.querySelector('[data-sensor="insulation"] .status-badge');
                if (insulationEl && insulationBadge) {
                    const isNormal = baseValues.insulation === 1;
                    insulationEl.textContent = isNormal ? '正常' : '异常';
                    insulationBadge.className = `badge ${isNormal ? 'bg-success' : 'bg-danger'} status-badge`;
                    insulationBadge.textContent = isNormal ? '正常' : '异常';
                }

                // 更新气象数据
                const calculateDewPoint = (temp, humidity) => {
                    // Magnus公式计算露点温度
                    const a = 17.27;
                    const b = 237.7;
                    const alpha = ((a * temp) / (b + temp)) + Math.log(humidity / 100);
                    return (b * alpha) / (a - alpha);
                };

                const calculateAirDensity = (pressure, temp) => {
                    // 理想气体方程计算空气密度
                    const R = 287.058; // 干燥空气的气体常数
                    return (pressure * 100) / (R * (temp + 273.15));
                };

                const calculateClearnessIndex = (measured, theoretical) => {
                    // 计算晴空指数
                    return Math.max(0, Math.min(1, measured / theoretical));
                };

                // 更新所有气象传感器数据
                const weatherSensors = {
                    'temperature': { value: updateValue(baseValues.envTemp, fluctuationRanges.envTemp), unit: '°C' },
                    'temp-gradient': { value: updateValue(baseValues.tempGradient, fluctuationRanges.tempGradient), unit: '°C/100m' },
                    'humidity': { value: updateValue(baseValues.humidity, fluctuationRanges.humidity), unit: '%' },
                    'dew-point': { value: calculateDewPoint(baseValues.envTemp, baseValues.humidity), unit: '°C' },
                    'wind-speed': { value: updateValue(baseValues.windSpeed, fluctuationRanges.windSpeed), unit: 'm/s' },
                    'wind-direction': { value: updateValue(baseValues.windDirection, fluctuationRanges.windDirection), unit: '°' },
                    'wind-shear': { value: updateValue(baseValues.windShear, fluctuationRanges.windShear), unit: 's⁻¹' },
                    'turbulence': { value: updateValue(baseValues.turbulence, fluctuationRanges.turbulence), unit: '' },
                    'visibility': { value: updateValue(baseValues.visibility, fluctuationRanges.visibility), unit: 'km' },
                    'pressure': { value: updateValue(baseValues.pressure, fluctuationRanges.pressure), unit: 'hPa' },
                    'air-density': { value: calculateAirDensity(baseValues.pressure, baseValues.envTemp), unit: 'kg/m³' },
                    'solar-radiation': { value: updateValue(baseValues.solarRadiation, fluctuationRanges.solarRadiation), unit: 'W/m²' },
                    'cloud-cover': { value: updateValue(baseValues.cloudCover, fluctuationRanges.cloudCover), unit: '%' },
                    'clearness-index': { value: updateValue(baseValues.clearnessIndex, fluctuationRanges.clearnessIndex), unit: '' },
                    'pm25': { value: updateValue(baseValues.pm25, fluctuationRanges.pm25), unit: 'µg/m³' },
                    'dust': { value: updateValue(baseValues.dust, fluctuationRanges.dust), unit: 'mg/m³' }
                };

                // 更新显示
                Object.entries(weatherSensors).forEach(([id, data]) => {
                    const el = document.querySelector(`[data-weather="${id}"] .sensor-value`);
                    if (el) {
                        el.textContent = `${Number(data.value).toFixed(2)}${data.unit}`;
                    }

                    // 更新状态标签
                    const threshold = weatherThresholds[id.replace('-', '')];
                    if (threshold) {
                        const badge = document.querySelector(`[data-weather="${id}"] .status-badge`);
                        if (badge) {
                            let className, text;
                            if (data.value >= threshold.high) {
                                className = 'bg-warning';
                                text = '偏高';
                            } else if (data.value <= threshold.low) {
                                className = 'bg-danger';
                                text = '偏低';
                            } else {
                                className = 'bg-success';
                                text = '正常';
                            }
                            badge.className = `badge ${className} status-badge`;
                            badge.textContent = text;
                        }
                    }
                });

                // 导出数据用于模型训练
                const timeSeriesData = {
                    timestamp: new Date().toISOString(),
                    location: {
                        latitude: this.droneMarker?.getLatLng().lat,
                        longitude: this.droneMarker?.getLatLng().lng,
                        altitude: baseValues.droneHeight
                    },
                    weather: Object.fromEntries(
                        Object.entries(weatherSensors).map(([key, data]) => [key, data.value])
                    ),
                    energy_related: {
                        solar_potential: weatherSensors['solar-radiation'].value * 
                            (1 - weatherSensors['cloud-cover'].value / 100) * 
                            weatherSensors['clearness-index'].value,
                        wind_potential: 0.5 * weatherSensors['air-density'].value * 
                            Math.pow(weatherSensors['wind-speed'].value, 3)
                    }
                };

                // 可以在这里添加数据导出逻辑
                console.log('Time series data for model training:', timeSeriesData);

            } catch (error) {
                console.error('Error updating weather data:', error);
            }
        }, 1000);

        // 存储定时器ID以便需要时清除
        this.updateInterval = updateInterval;
    }

    // 初始化全屏功能
    initializeFullscreen() {
        // 绑定所有全屏按钮的事件
        document.querySelectorAll('.fullscreen-btn').forEach(button => {
            button.addEventListener('click', () => {
                const viewType = button.dataset.view;
                this.enterFullscreen(viewType);
            });
        });

        // 初始化全屏容器
        this.fullscreenContainer = document.getElementById('fullscreen-container');
        this.fullscreenMap = document.getElementById('fullscreen-map');
        this.fullscreenVideo = document.getElementById('fullscreen-video');
        this.fullscreenThermal = document.getElementById('fullscreen-thermal');

        // 绑定视图切换按钮事件
        const viewButtons = document.querySelectorAll('.view-switcher .btn');
        viewButtons.forEach(button => {
            button.addEventListener('click', () => {
                const view = button.dataset.view;
                this.switchFullscreenView(view);
            });
        });

        // 绑定退出全屏按钮事件
        document.getElementById('exit-fullscreen').addEventListener('click', () => {
            this.exitFullscreen();
        });

        // 初始化视频播放
        this.initializeVideoPlayers();
    }

    // 初始化视频播放器
    initializeVideoPlayers() {
        // 普通视图的视频播放器
        this.normalVideoPlayer = document.getElementById('video-player');
        if (this.normalVideoPlayer) {
            this.normalVideoPlayer.play().catch(error => {
                console.warn('Auto-play failed:', error);
            });
        }

        // 全屏视图的视频播放器
        this.fullscreenVideoPlayer = document.getElementById('fullscreen-video-player');
    }

    // 进入全屏模式
    enterFullscreen(viewId) {
        try {
            // 显示全屏容器
            this.fullscreenContainer.classList.add('active');
            document.body.style.overflow = 'hidden';

            // 保存当前视频播放时间（如果在播放视频）
            if (viewId === 'video' && this.normalVideoPlayer) {
                this.videoCurrentTime = this.normalVideoPlayer.currentTime;
            }

            // 立即切换到对应视图
            this.switchFullscreenView(viewId);

            // 确保地图在全屏模式下正确初始化
            if (viewId === 'map') {
                setTimeout(() => {
                    this.initializeFullscreenMap();
                }, 100);
            }
        } catch (error) {
            console.error('Error entering fullscreen:', error);
        }
    }

    // 切换全屏视图
    switchFullscreenView(viewId) {
        try {
            // 隐藏所有视图
            this.fullscreenMap.classList.remove('active');
            this.fullscreenVideo.classList.remove('active');
            this.fullscreenThermal.classList.remove('active');

            // 移除之前的地图实例
            if (this.fullscreenMapInstance) {
                this.fullscreenMapInstance.remove();
                this.fullscreenMapInstance = null;
            }

            // 显示选中的视图
            switch(viewId) {
                case 'map':
                    this.fullscreenMap.classList.add('active');
                    this.initializeFullscreenMap();
                    break;
                    
                case 'video':
                    this.fullscreenVideo.classList.add('active');
                    if (this.fullscreenVideoPlayer) {
                        this.fullscreenVideoPlayer.src = 'static/video/drone_inspection.mp4';
                        // 恢复之前的播放位置
                        if (this.videoCurrentTime) {
                            this.fullscreenVideoPlayer.currentTime = this.videoCurrentTime;
                        }
                        this.fullscreenVideoPlayer.play().catch(error => {
                            console.warn('Auto-play failed:', error);
                        });
                    }
                    break;
                    
                case 'thermal':
                    this.fullscreenThermal.classList.add('active');
                    const thermalContainer = document.getElementById('fullscreen-thermal');
                    if (thermalContainer) {
                        thermalContainer.innerHTML = '';
                        const thermalImg = document.createElement('img');
                        thermalImg.src = 'static/image/solar_panels_it_image.png';
                        thermalImg.style.width = '100%';
                        thermalImg.style.height = '100%';
                        thermalImg.style.objectFit = 'contain';
                        thermalContainer.appendChild(thermalImg);
                    }
                    break;
            }

            // 更新按钮状态
            const buttons = document.querySelectorAll('.view-switcher .btn');
            buttons.forEach(button => {
                if (button.dataset.view === viewId) {
                    button.classList.add('active');
                    button.classList.remove('btn-outline-light');
                    button.classList.add('btn-light');
                } else {
                    button.classList.remove('active');
                    button.classList.add('btn-outline-light');
                    button.classList.remove('btn-light');
                }
            });
        } catch (error) {
            console.error('Error switching fullscreen view:', error);
        }
    }

    // 退出全屏模式
    exitFullscreen() {
        try {
            // 如果正在播放视频，保存当前播放位置
            if (this.fullscreenVideoPlayer && !this.fullscreenVideoPlayer.paused) {
                this.videoCurrentTime = this.fullscreenVideoPlayer.currentTime;
            }

            // 停止全屏视频播放
            if (this.fullscreenVideoPlayer) {
                this.fullscreenVideoPlayer.pause();
                this.fullscreenVideoPlayer.src = '';
            }

            // 恢复普通视图的视频播放
            if (this.normalVideoPlayer && this.videoCurrentTime) {
                this.normalVideoPlayer.currentTime = this.videoCurrentTime;
                this.normalVideoPlayer.play().catch(error => {
                    console.warn('Auto-play failed:', error);
                });
            }

            // 隐藏全屏容器
            this.fullscreenContainer.classList.remove('active');
            document.body.style.overflow = '';
            
            // 如果存在全屏地图实例，移除它
            if (this.fullscreenMapInstance) {
                this.fullscreenMapInstance.remove();
                this.fullscreenMapInstance = null;
            }
            
            // 重置所有视图
            this.fullscreenMap.classList.remove('active');
            this.fullscreenVideo.classList.remove('active');
            this.fullscreenThermal.classList.remove('active');

            // 清理热成像视图
            const thermalContainer = document.getElementById('fullscreen-thermal');
            if (thermalContainer) {
                thermalContainer.innerHTML = '';
            }
        } catch (error) {
            console.error('Error exiting fullscreen:', error);
        }
    }

    // 初始化图表
    initializeCharts() {
        try {
            // 检查是否已经初始化
            if (this.chartsInitialized) {
                return;
            }

            // 清理已存在的图表实例
            this.cleanupCharts();

            // 初始化各个图表
            const charts = {
                temperatureHeatmap: "#temperatureHeatmap",
                efficiencyTrend: "#efficiencyTrend",
                tempEfficiencyChart: "#tempEfficiencyChart",
                windDustChart: "#windDustChart",
                anomalyPieChart: "#anomalyPieChart"
            };

            // 检查每个图表容器是否存在，并且只初始化一次
            Object.entries(charts).forEach(([chartName, containerId]) => {
                const container = document.querySelector(containerId);
                if (container && !container.hasAttribute('data-chart-initialized')) {
                    switch(chartName) {
                        case 'temperatureHeatmap':
                            this.initTemperatureHeatmap();
                            break;
                        case 'efficiencyTrend':
                            this.initEfficiencyChart();
                            break;
                        case 'tempEfficiencyChart':
                        case 'windDustChart':
                            if (chartName === 'tempEfficiencyChart') {
                                this.initWeatherCorrelationCharts();
                            }
                            break;
                        case 'anomalyPieChart':
                            this.initAnomalyPieChart();
                            break;
                    }
                    // 标记容器已初始化
                    container.setAttribute('data-chart-initialized', 'true');
                }
            });

            // 开始定期更新图表数据
            this.startChartUpdates();

            // 标记图表已初始化
            this.chartsInitialized = true;
        } catch (error) {
            console.error('Error initializing charts:', error);
        }
    }

    // 清理图表实例
    cleanupCharts() {
        const charts = [
            'temperatureChart',
            'efficiencyChart',
            'tempEfficiencyCorrelation',
            'windDustCorrelation',
            'anomalyPieChart',
            'healthTrendChart'
        ];

        charts.forEach(chartName => {
            if (this[chartName]) {
                try {
                    this[chartName].destroy();
                    this[chartName] = null;
                } catch (error) {
                    console.error(`Error destroying ${chartName}:`, error);
                }
            }
        });

        // 清除所有图表初始化标记
        document.querySelectorAll('[data-chart-initialized]').forEach(el => {
            el.removeAttribute('data-chart-initialized');
        });
    }

    // 初始化温度分布热力图
    initTemperatureHeatmap() {
        const container = document.querySelector("#temperatureHeatmap");
        if (!container) return;

        const generateData = (count, yrange) => {
            let series = [];
            for(let i = 0; i < count; i++) {
                series.push({
                    x: `区域 ${String.fromCharCode(65 + Math.floor(i/3))}${(i%3)+1}`,
                    y: (Math.random() * (yrange.max - yrange.min) + yrange.min).toFixed(1)
                });
            }
            return series;
        };

        const options = {
            series: [{
                name: '表面温度',
                data: generateData(9, { min: 25, max: 75 })
            }],
            chart: {
                height: 200,
                type: 'heatmap',
                toolbar: { show: false }
            },
            plotOptions: {
                heatmap: {
                    shadeIntensity: 0.5,
                    colorScale: {
                        ranges: [{
                            from: 25,
                            to: 35,
                            name: '正常',
                            color: '#00A100'
                        },{
                            from: 35,
                            to: 45,
                            name: '注意',
                            color: '#FFB200'
                        },{
                            from: 45,
                            to: 75,
                            name: '警告',
                            color: '#FF0000'
                        }]
                    }
                }
            },
            dataLabels: { enabled: true },
            title: { text: '光伏板温度分布 (°C)' }
        };

        this.temperatureChart = new ApexCharts(container, options);
        this.temperatureChart.render();
    }

    // 更新图表数据
    updateChartData() {
        try {
            // 更新温度分布热力图
            if (this.temperatureChart) {
                const newTempData = Array(9).fill(0).map(() => ({
                    x: `区域 ${String.fromCharCode(65 + Math.floor(Math.random() * 3))}${Math.floor(Math.random() * 3) + 1}`,
                    y: (Math.random() * (75 - 25) + 25).toFixed(1)
                }));
                this.temperatureChart.updateSeries([{
                    data: newTempData
                }]);
            }

            // 更新发电效率趋势
            if (this.efficiencyChart) {
                const newEfficiencyData = Array(24).fill(0).map(() => 
                    (Math.random() * (0.95 - 0.75) + 0.75).toFixed(2)
                );
                this.efficiencyChart.updateSeries([{
                    data: newEfficiencyData
                }]);
            }

            // 更新异常分布
            if (this.anomalyPieChart) {
                this.anomalyPieChart.updateSeries([
                    Math.floor(Math.random() * 20 + 35),
                    Math.floor(Math.random() * 15 + 20),
                    Math.floor(Math.random() * 10 + 15),
                    Math.floor(Math.random() * 10 + 5)
                ]);
            }
        } catch (error) {
            console.error('Error updating chart data:', error);
        }
    }

    // 初始化温度分布热力图
    initEfficiencyChart() {
        const options = {
            series: [{
                name: '发电效率',
                data: Array(24).fill(0).map(() => 
                    (Math.random() * (0.95 - 0.75) + 0.75).toFixed(2)
                )
            }],
            chart: {
                height: 200,
                type: 'line',
                toolbar: { show: false },
                animations: {
                    enabled: true,
                    easing: 'linear',
                    dynamicAnimation: {
                        speed: 1000
                    }
                }
            },
            stroke: {
                curve: 'smooth',
                width: 3
            },
            xaxis: {
                categories: Array(24).fill(0).map((_, i) => `${i}:00`)
            },
            yaxis: {
                min: 0.7,
                max: 1,
                tickAmount: 3,
                labels: {
                    formatter: (val) => `${(val * 100).toFixed(0)}%`
                }
            },
            markers: {
                size: 4,
                colors: ['#2196F3'],
                strokeWidth: 2,
                hover: { size: 6 }
            },
            tooltip: {
                y: {
                    formatter: (val) => `${(val * 100).toFixed(1)}%`
                }
            }
        };

        this.efficiencyChart = new ApexCharts(document.querySelector("#efficiencyTrend"), options);
        this.efficiencyChart.render();
    }

    // 初始化气象相关性图表
    initWeatherCorrelationCharts() {
        // 温度-发电效率相关性图表
        const tempEffOptions = {
            series: [{
                name: '发电效率',
                data: Array(10).fill(0).map(() => ({
                    x: (Math.random() * (40 - 20) + 20).toFixed(1),
                    y: (Math.random() * (0.95 - 0.75) + 0.75).toFixed(2)
                }))
            }],
            chart: {
                height: 100,
                type: 'scatter',
                toolbar: { show: false }
            },
            xaxis: {
                title: { text: '温度 (°C)' },
                tickAmount: 5
            },
            yaxis: {
                title: { text: '效率' },
                labels: {
                    formatter: (val) => `${(val * 100).toFixed(0)}%`
                }
            }
        };

        this.tempEfficiencyCorrelation = new ApexCharts(
            document.querySelector("#tempEfficiencyChart"), 
            tempEffOptions
        );
        this.tempEfficiencyCorrelation.render();

        // 风速-清洁度相关性图表
        const windDustOptions = {
            series: [{
                name: '清洁度',
                data: Array(10).fill(0).map(() => ({
                    x: (Math.random() * (10 - 0)).toFixed(1),
                    y: (Math.random() * (100 - 70) + 70).toFixed(1)
                }))
            }],
            chart: {
                height: 100,
                type: 'scatter',
                toolbar: { show: false }
            },
            xaxis: {
                title: { text: '风速 (m/s)' },
                tickAmount: 5
            },
            yaxis: {
                title: { text: '清洁度' },
                labels: {
                    formatter: (val) => `${val.toFixed(0)}%`
                }
            }
        };

        this.windDustCorrelation = new ApexCharts(
            document.querySelector("#windDustChart"), 
            windDustOptions
        );
        this.windDustCorrelation.render();
    }

    // 初始化异常类型分布饼图
    initAnomalyPieChart() {
        const container = document.querySelector("#anomalyPieChart");
        if (!container || container.hasAttribute('data-chart-initialized')) {
            return;
        }

        const options = {
            series: [44, 28, 18, 10],
            chart: {
                height: 150,
                type: 'donut',
                toolbar: { show: false }
            },
            labels: ['温度异常', '效率降低', '污渍覆盖', '其他'],
            colors: ['#FF4081', '#FFC107', '#2196F3', '#9E9E9E'],
            legend: {
                position: 'bottom',
                horizontalAlign: 'center',
                fontSize: '12px'
            },
            plotOptions: {
                pie: {
                    donut: {
                        size: '70%'
                    }
                }
            }
        };

        if (this.anomalyPieChart) {
            this.anomalyPieChart.destroy();
        }

        this.anomalyPieChart = new ApexCharts(container, options);
        this.anomalyPieChart.render();
        container.setAttribute('data-chart-initialized', 'true');
    }

    // 初始化设备健康度趋势图
    initHealthTrendChart() {
        const options = {
            series: [{
                name: '健康度',
                data: Array(7).fill(0).map(() => 
                    (Math.random() * (98 - 85) + 85).toFixed(1)
                )
            }],
            chart: {
                height: 150,
                type: 'area',
                toolbar: { show: false },
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
            xaxis: {
                categories: ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            },
            yaxis: {
                min: 80,
                max: 100,
                labels: {
                    formatter: (val) => `${val.toFixed(0)}%`
                }
            },
            colors: ['#4CAF50']
        };

        this.healthTrendChart = new ApexCharts(document.querySelector("#healthTrendChart"), options);
        this.healthTrendChart.render();
    }

    // 开始定期更新图表数据
    startChartUpdates() {
        // 每分钟更新一次图表数据
        setInterval(() => {
            this.updateChartData();
        }, 60000);

        // 如果实时更新开关打开，则更频繁地更新相关性图表
        const realtimeUpdateSwitch = document.getElementById('realtimeUpdate');
        if (realtimeUpdateSwitch) {
            realtimeUpdateSwitch.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startRealtimeUpdates();
                } else {
                    this.stopRealtimeUpdates();
                }
            });
        }
    }

    // 开始实时更新
    startRealtimeUpdates() {
        this.realtimeUpdateInterval = setInterval(() => {
            // 更新相关性图表数据
            const newTempEffData = Array(10).fill(0).map(() => ({
                x: (Math.random() * (40 - 20) + 20).toFixed(1),
                y: (Math.random() * (0.95 - 0.75) + 0.75).toFixed(2)
            }));
            this.tempEfficiencyCorrelation.updateSeries([{
                data: newTempEffData
            }]);

            const newWindDustData = Array(10).fill(0).map(() => ({
                x: (Math.random() * 10).toFixed(1),
                y: (Math.random() * (100 - 70) + 70).toFixed(1)
            }));
            this.windDustCorrelation.updateSeries([{
                data: newWindDustData
            }]);
        }, 5000); // 每5秒更新一次
    }

    // 停止实时更新
    stopRealtimeUpdates() {
        if (this.realtimeUpdateInterval) {
            clearInterval(this.realtimeUpdateInterval);
        }
    }

    // 初始化全屏地图
    initializeFullscreenMap() {
        try {
            // 检查原始地图是否存在
            if (!this.map) {
                console.error('Original map is not initialized.');
                return;
            }

            // 如果全屏地图已经存在，先移除它
            if (this.fullscreenMapInstance) {
                this.fullscreenMapInstance.remove();
            }

            // 创建新的地图实例
            this.fullscreenMapInstance = L.map('fullscreen-map', {
                center: this.map.getCenter(),
                zoom: this.map.getZoom(),
                zoomControl: true
            });

            // 添加卫星图层
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            }).addTo(this.fullscreenMapInstance);

            // 复制原始地图的图层到全屏地图
            this.map.eachLayer((layer) => {
                if (layer instanceof L.Marker || layer instanceof L.Polygon || layer instanceof L.Polyline) {
                    // 克隆图层以避免引用问题
                    const clonedLayer = this.cloneLayer(layer);
                    if (clonedLayer) {
                        clonedLayer.addTo(this.fullscreenMapInstance);
                    }
                }
            });

            // 立即触发地图重绘
            this.fullscreenMapInstance.invalidateSize();

            // 添加一个短暂延迟后再次触发重绘，以确保地图正确显示
            setTimeout(() => {
                this.fullscreenMapInstance.invalidateSize();
            }, 200);
        } catch (error) {
            console.error('Error initializing fullscreen map:', error);
        }
    }

    // 克隆地图图层
    cloneLayer(layer) {
        try {
            if (layer instanceof L.Marker) {
                return L.marker(layer.getLatLng(), {
                    icon: layer.options.icon,
                    zIndexOffset: layer.options.zIndexOffset
                });
            } else if (layer instanceof L.Polygon) {
                return L.polygon(layer.getLatLngs(), {
                    color: layer.options.color,
                    weight: layer.options.weight,
                    fillColor: layer.options.fillColor,
                    fillOpacity: layer.options.fillOpacity
                });
            } else if (layer instanceof L.Polyline) {
                return L.polyline(layer.getLatLngs(), {
                    color: layer.options.color,
                    weight: layer.options.weight,
                    opacity: layer.options.opacity,
                    dashArray: layer.options.dashArray
                });
            }
        } catch (error) {
            console.error('Error cloning layer:', error);
            return null;
        }
    }

    // 预览巡检报告
    previewInspectionReport() {
        try {
            // 检查必要的DOM元素
            const previewModal = document.getElementById('reportPreviewModal');
            const previewContent = document.getElementById('reportPreviewContent');
            
            if (!previewModal || !previewContent) {
                console.warn('Preview modal elements not found');
                return; // 静默失败，不显示错误提示
            }
            
            // 收集报告数据
            const reportData = this.collectReportData();
            if (!reportData) {
                console.warn('Failed to collect report data');
                return; // 静默失败，不显示错误提示
            }
            
            // 生成报告HTML
            const reportHtml = this.generateReportHTML(reportData);
            
            // 更新预览内容
            previewContent.innerHTML = reportHtml;
            
            // 显示预览模态框
            const bsModal = new bootstrap.Modal(previewModal);
            bsModal.show();
            
        } catch (error) {
            // 只在开发环境中记录错误
            console.debug('Preview generation debug info:', error);
            // 不向用户显示错误提示
        }
    }
}

// 初始化系统
document.addEventListener('DOMContentLoaded', () => {
    window.droneSystem = new DroneInspectionSystem();
}); 