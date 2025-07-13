const devices = {
    // 成都光伏电站 - 位于双流区太阳能光伏产业园
    'CD001': {
        id: 'CD001',
        name: '1号光伏板阵列',
        type: 'solar',
        serialNumber: 'SPV20230001',
        manufacturer: '隆基绿能',
        model: 'Hi-MO 6 80K',
        installDate: '2023-01-15',
        productionDate: '2022-12-01',
        warrantyPeriod: '2033-01-15',
        area: 'chengdu',
        capacity: '100MW',
        location: {
            longitude: '103.923544',
            latitude: '30.574673',
            altitude: '485.5',
            installationSite: '成都市双流区太阳能光伏产业园',
            building: 'A3栋',
            floor: '楼顶',
            room: '露天',
            gpsAccuracy: '±2m'
        },
        status: 'running',
        health: 95,
        remainingLife: 320,
        lastMaintenance: '2023-12-15',
        nextMaintenance: '2024-03-15',
        maintenanceCycle: 90, // 天
        operatingHours: 8760,
        downtime: 24,
        mttf: 8760, // 平均故障间隔时间(小时)
        mttr: 4, // 平均修复时间(小时)
        params: {
            tiltAngle: '32°',
            orientation: '正南',
            efficiency: '21.3%',
            dailyOutput: '432.5MWh',
            temperature: '42.3°C',
            cleanStatus: '良好',
            panelType: '单晶硅',
            cellEfficiency: '23.1%',
            moduleArea: '2.384m²',
            moduleCount: 50000,
            bypassDiodes: 3,
            temperatureCoefficient: '-0.35%/°C',
            nominalOperatingTemp: '45±2°C',
            maximumSystemVoltage: '1500V',
            shortCircuitCurrent: '13.9A',
            openCircuitVoltage: '45.5V'
        },
        currentPower: '78.5MW',
        efficiency: '98.2%',
        dailyGeneration: '432.5MWh',
        monthlyGeneration: '12975MWh',
        yearlyGeneration: '157000MWh',
        totalGeneration: '256780MWh',
        co2Reduction: '128390t',
        maintenanceRecords: [
            {
                date: '2023-12-15',
                type: 'regular',
                description: '季度常规检修',
                findings: '清洁度良好，连接正常',
                actions: '清洁面板，紧固连接件',
                parts: '无更换',
                technician: '张工',
                cost: 5000,
                nextDate: '2024-03-15'
            }
        ],
        alarmRecords: [
            {
                timestamp: '2024-01-20T14:30:00',
                type: 'warning',
                code: 'W001',
                description: '发电效率轻微下降',
                value: '92%',
                threshold: '95%',
                status: 'resolved',
                resolveTime: '2024-01-20T16:30:00',
                resolution: '清洁光伏板表面'
            }
        ],
        documents: {
            manual: '/docs/CD001/manual.pdf',
            warranty: '/docs/CD001/warranty.pdf',
            certification: '/docs/CD001/certification.pdf',
            installationGuide: '/docs/CD001/installation.pdf',
            maintenanceGuide: '/docs/CD001/maintenance.pdf'
        },
        spareParts: [
            {
                code: 'SP001',
                name: '光伏板',
                model: 'Hi-MO 6 80K',
                quantity: 50,
                minimumStock: 20,
                supplier: '隆基绿能',
                lastPurchase: '2023-06-15',
                unitPrice: 1200
            }
        ],
        inspectionData: {
            lastInspection: '2024-01-15',
            nextInspection: '2024-02-15',
            inspectionCycle: 30,
            thermalImaging: {
                date: '2024-01-15',
                maxTemp: '45.2°C',
                minTemp: '38.6°C',
                hotspots: 0
            },
            electricalTests: {
                date: '2024-01-15',
                insulationResistance: '500MΩ',
                groundingResistance: '0.1Ω'
            }
        },
        assetInfo: {
            purchaseDate: '2022-12-15',
            purchasePrice: 45000000,
            depreciation: {
                method: '直线折旧',
                period: 20,
                salvageValue: 2250000,
                currentValue: 42750000
            },
            insurance: {
                provider: '中国人保',
                policyNumber: 'INS20230001',
                coverage: 50000000,
                expiryDate: '2024-12-31'
            }
        },
        energyAnalysis: {
            designedCapacity: '100MW',
            actualCapacity: '98.5MW',
            capacityFactor: '85%',
            performanceRatio: '0.82',
            specificYield: '1580kWh/kWp',
            gridIntegration: {
                gridVoltage: '35kV',
                powerFactor: '0.99',
                harmonicDistortion: '2.1%'
            }
        }
    },
    'CD002': {
        id: 'CD002',
        name: '2号逆变器组',
        type: 'solar',
        serialNumber: 'INV20230002',
        manufacturer: '阳光电源',
        installDate: '2023-01-15',
        area: 'chengdu',
        capacity: '50MW',
        location: {
            longitude: '103.924544',
            latitude: '30.575673',
            altitude: '485.5',
            installationSite: '成都市双流区太阳能光伏产业园'
        },
        status: 'maintenance',
        health: 82,
        remainingLife: 280,
        params: {
            inputVoltage: '800V',
            outputVoltage: '380V',
            efficiency: '98.9%',
            temperature: '38.5°C',
            loadRate: '85%'
        },
        currentPower: '42.3MW',
        efficiency: '97.8%'
    },
    'CD003': {
        id: 'CD003',
        name: '3号逆变器阵列',
        type: 'solar',
        serialNumber: 'INV20230003',
        manufacturer: '阳光电源',
        model: 'SG250HX',
        installDate: '2023-01-15',
        productionDate: '2022-12-01',
        warrantyPeriod: '2033-01-15',
        area: 'chengdu',
        capacity: '75MW',
        location: {
            longitude: '103.925544',
            latitude: '30.576673',
            altitude: '485.5',
            installationSite: '成都市双流区太阳能光伏产业园C区',
            building: 'C2栋',
            floor: '1层',
            room: 'C2-101',
            gpsAccuracy: '±2m'
        },
        status: 'running',
        health: 98,
        remainingLife: 355,
        params: {
            inputVoltage: '850V',
            outputVoltage: '380V',
            efficiency: '99.1%',
            temperature: '35.2°C',
            loadRate: '92%',
            powerFactor: '0.99',
            dcInputs: 12,
            mpptChannels: 6,
            maxEfficiency: '99.3%',
            europeanEfficiency: '98.9%',
            coolingMethod: '智能风冷',
            protectionLevel: 'IP66',
            communicationProtocol: 'Modbus-RTU'
        },
        currentPower: '68.5MW',
        efficiency: '99.1%',
        maintenanceRecords: [
            {
                date: '2023-12-20',
                type: 'regular',
                description: '季度例行检查',
                findings: '运行正常，散热系统正常',
                actions: '清洁散热器，更新软件',
                parts: '无更换',
                technician: '李工',
                cost: 2000,
                nextDate: '2024-03-20'
            }
        ],
        alarmRecords: [
            {
                timestamp: '2024-01-15T08:30:00',
                type: 'info',
                code: 'I001',
                description: '系统自动更新完成',
                value: '-',
                threshold: '-',
                status: 'resolved',
                resolveTime: '2024-01-15T08:35:00',
                resolution: '自动完成'
            }
        ]
    },

    // 攀枝花光伏电站 - 位于仁和区光照资源丰富区域
    'PZH001': {
        id: 'PZH001',
        name: '1号光伏方阵',
        type: 'solar',
        serialNumber: 'SPV20230003',
        manufacturer: '晶科能源',
        installDate: '2023-03-20',
        area: 'panzhihua',
        capacity: '150MW',
        location: {
            longitude: '101.738637',
            latitude: '26.502108',
            altitude: '1200',
            installationSite: '攀枝花市仁和区太阳能发电基地'
        },
        status: 'running',
        health: 97,
        remainingLife: 350,
        params: {
            tiltAngle: '28°',
            orientation: '正南偏东5°',
            efficiency: '22.1%',
            dailyOutput: '658.3MWh',
            temperature: '45.2°C',
            cleanStatus: '良好'
        },
        currentPower: '142.8MW',
        efficiency: '99.1%'
    },

    // 雅安风电场 - 位于雨城区周公山风电场
    'YA001': {
        id: 'YA001',
        name: '1号风力发电机组',
        type: 'wind',
        serialNumber: 'WTG20230001',
        manufacturer: '金风科技',
        installDate: '2023-02-10',
        area: 'yaan',
        capacity: '3.6MW',
        location: {
            longitude: '103.082033',
            latitude: '29.987515',
            altitude: '2100',
            installationSite: '雅安市雨城区周公山风电场'
        },
        status: 'running',
        health: 93,
        remainingLife: 310,
        params: {
            hubHeight: '90m',
            rotorDiameter: '140m',
            windSpeed: '8.5m/s',
            rotationSpeed: '14.2rpm',
            pitch: '15°',
            direction: '东南'
        },
        currentPower: '3.2MW',
        efficiency: '96.5%'
    },
    'YA002': {
        id: 'YA002',
        name: '2号风力发电机组',
        type: 'wind',
        serialNumber: 'WTG20230002',
        manufacturer: '金风科技',
        installDate: '2023-02-10',
        area: 'yaan',
        capacity: '3.6MW',
        location: {
            longitude: '103.083033',
            latitude: '29.988515',
            altitude: '2100',
            installationSite: '雅安市雨城区周公山风电场'
        },
        status: 'fault',
        health: 65,
        remainingLife: 180,
        params: {
            hubHeight: '90m',
            rotorDiameter: '140m',
            windSpeed: '7.8m/s',
            rotationSpeed: '0rpm',
            pitch: '90°',
            direction: '东南'
        },
        currentPower: '0MW',
        efficiency: '0%'
    },
    'YA003': {
        id: 'YA003',
        name: '3号智能风机',
        type: 'wind',
        serialNumber: 'WTG20230003',
        manufacturer: '远景能源',
        model: 'EN-171/6.7MW',
        installDate: '2023-06-01',
        productionDate: '2023-04-15',
        warrantyPeriod: '2033-06-01',
        area: 'yaan',
        capacity: '6.7MW',
        location: {
            longitude: '103.084033',
            latitude: '29.989515',
            altitude: '2150',
            installationSite: '雅安市雨城区周公山风电场C区'
        },
        status: 'running',
        health: 97,
        remainingLife: 340,
        params: {
            hubHeight: '100m',
            rotorDiameter: '171m',
            windSpeed: '12.5m/s',
            rotationSpeed: '9.8rpm',
            pitch: '15°',
            direction: '东南',
            cutInWindSpeed: '3m/s',
            cutOutWindSpeed: '25m/s',
            ratedWindSpeed: '11.5m/s',
            generatorType: '永磁直驱',
            gridVoltage: '35kV',
            bladeMaterial: '碳纤维复合材料'
        },
        currentPower: '5.8MW',
        efficiency: '98.2%',
        maintenanceRecords: [
            {
                date: '2024-01-10',
                type: 'preventive',
                description: '叶片检查和维护',
                findings: '叶片状态良好，需要清洁',
                actions: '清洁叶片，检查螺栓紧固',
                parts: '无更换',
                technician: '张工',
                cost: 8000,
                nextDate: '2024-04-10'
            }
        ]
    },

    // 广元储能电站 - 位于利州区新能源产业园
    'GY001': {
        id: 'GY001',
        name: '1号储能系统',
        type: 'storage',
        serialNumber: 'ESS20230001',
        manufacturer: '宁德时代',
        installDate: '2023-04-01',
        area: 'guangyuan',
        capacity: '100MWh',
        location: {
            longitude: '105.819687',
            latitude: '32.433668',
            altitude: '550',
            installationSite: '广元市利州区新能源产业园'
        },
        status: 'running',
        health: 96,
        remainingLife: 340,
        params: {
            batteryType: '磷酸铁锂',
            chargeStatus: '85%',
            temperature: '25.6°C',
            cycleCount: '126',
            chargeRate: '0.5C',
            dischargeRate: '1C'
        },
        currentPower: '45MW',
        efficiency: '98.5%'
    },
    'GY002': {
        id: 'GY002',
        name: '2号储能变流器',
        type: 'storage',
        serialNumber: 'PCS20230002',
        manufacturer: '阳光电源',
        model: 'SC2000UD',
        installDate: '2023-07-15',
        productionDate: '2023-06-01',
        warrantyPeriod: '2033-07-15',
        area: 'guangyuan',
        capacity: '2000kW',
        location: {
            longitude: '105.820687',
            latitude: '32.434668',
            altitude: '552',
            installationSite: '广元市利州区新能源产业园B区'
        },
        status: 'running',
        health: 96,
        remainingLife: 330,
        params: {
            ratedPower: '2000kW',
            ratedVoltage: '1000V',
            gridFrequency: '50Hz',
            efficiency: '98.9%',
            powerFactor: '0.99',
            thd: '<3%',
            coolingMethod: '液冷',
            protectionLevel: 'IP65',
            operatingTemp: '-20~50°C',
            altitude: '≤4000m'
        },
        currentPower: '1850kW',
        efficiency: '98.7%'
    },

    // 德阳油气田 - 位于旌阳区
    'DY001': {
        id: 'DY001',
        name: '1号采油机',
        type: 'oil',
        serialNumber: 'OPU20230001',
        manufacturer: '杰瑞股份',
        installDate: '2023-01-05',
        area: 'deyang',
        capacity: '150t/d',
        location: {
            longitude: '104.398651',
            latitude: '31.127991',
            altitude: '500',
            installationSite: '德阳市旌阳区能源基地'
        },
        status: 'running',
        health: 88,
        remainingLife: 275,
        params: {
            pumpDepth: '2150m',
            strokeLength: '3m',
            strokePerMinute: '4.2',
            wellheadPressure: '2.5MPa',
            motorTemp: '48.2°C',
            oilPressure: '12.5MPa'
        },
        currentOutput: '142t/d',
        efficiency: '94.7%'
    },
    'DY002': {
        id: 'DY002',
        name: '2号智能采油机',
        type: 'oil',
        serialNumber: 'OPU20230002',
        manufacturer: '杰瑞股份',
        model: 'JR-AI-8000',
        installDate: '2023-08-20',
        productionDate: '2023-07-05',
        warrantyPeriod: '2033-08-20',
        area: 'deyang',
        capacity: '200t/d',
        location: {
            longitude: '104.399651',
            latitude: '31.128991',
            altitude: '502',
            installationSite: '德阳市旌阳区能源基地B区'
        },
        status: 'running',
        health: 95,
        remainingLife: 320,
        params: {
            pumpDepth: '2300m',
            strokeLength: '3.5m',
            strokePerMinute: '4.5',
            wellheadPressure: '2.8MPa',
            motorTemp: '45.5°C',
            oilPressure: '13.2MPa',
            pumpEfficiency: '92%',
            motorPower: '75kW',
            controlMode: 'AI智能优化',
            balanceAccuracy: '98.5%'
        },
        currentOutput: '185t/d',
        efficiency: '95.2%'
    }
};

// 区域信息
const areas = {
    chengdu: {
        name: '成都光伏电站',
        totalCapacity: '500MW',
        location: '成都市双流区',
        deviceCount: 15,
        coordinates: {
            longitude: '103.923544',
            latitude: '30.574673'
        },
        weather: {
            annualSunHours: 1150,
            averageTemperature: '16.5°C',
            averageHumidity: '80%',
            annualRainfall: '900mm'
        },
        gridConnection: {
            voltage: '220kV',
            substation: '双流变电站',
            transmissionLine: '成双线',
            transformerCapacity: '600MVA',
            powerFactor: '0.98',
            gridStability: '99.99%'
        },
        staff: {
            operators: 25,
            technicians: 15,
            managers: 5,
            security: 8
        },
        certifications: [
            'ISO 14001:2015',
            'ISO 45001:2018',
            'ISO 50001:2018',
            'ISO 9001:2015'
        ],
        facilities: {
            controlRoom: '中央控制室',
            warehouse: '设备仓库',
            office: '管理办公室',
            laboratory: '设备检测实验室',
            trainingCenter: '培训中心'
        }
    },
    panzhihua: {
        name: '攀枝花光伏电站',
        totalCapacity: '800MW',
        location: '攀枝花市仁和区',
        deviceCount: 25
    },
    yaan: {
        name: '雅安风电场',
        totalCapacity: '100MW',
        location: '雅安市雨城区',
        deviceCount: 28
    },
    guangyuan: {
        name: '广元储能电站',
        totalCapacity: '500MWh',
        location: '广元市利州区',
        deviceCount: 12
    },
    deyang: {
        name: '德阳油气田',
        totalCapacity: '2000t/d',
        location: '德阳市旌阳区',
        deviceCount: 35
    }
};

// 设备类型信息
const deviceTypes = {
    solar: {
        name: '光伏设备',
        icon: 'solar-panel',
        color: '#f1c40f',
        category: '发电设备',
        maintenanceGuidelines: {
            inspectionInterval: 30,
            cleaningInterval: 90,
            majorMaintenanceInterval: 365,
            specialRequirements: [
                '防尘防水检查',
                '接线盒密封性检查',
                '支架结构检查',
                '绝缘性能测试'
            ]
        },
        standardParameters: {
            normalEfficiencyRange: '15-23%',
            normalTemperatureRange: '20-60°C',
            normalPowerFactorRange: '0.95-1.0',
            normalVoltageRange: '600-1000V'
        },
        requiredQualifications: [
            '光伏系统运维工程师',
            '电气工程师',
            '安全工程师'
        ],
        safetyProcedures: [
            '高压电气操作规程',
            '高空作业安全规程',
            '防雷接地检查规程'
        ]
    },
    wind: {
        name: '风电设备',
        icon: 'fan',
        color: '#3498db'
    },
    storage: {
        name: '储能设备',
        icon: 'battery',
        color: '#2ecc71'
    },
    oil: {
        name: '油气设备',
        icon: 'oil-pump',
        color: '#e74c3c'
    }
};

// 状态定义
const statusTypes = {
    running: {
        name: '正常运行',
        color: 'success',
        code: 1,
        description: '设备正常运行中',
        requiredActions: [],
        alertLevel: 'normal'
    },
    maintenance: {
        name: '停机维护',
        color: 'warning',
        code: 2,
        description: '设备处于计划维护状态',
        requiredActions: ['维护记录', '备件检查'],
        alertLevel: 'warning'
    },
    fault: {
        name: '故障停机',
        color: 'danger',
        code: 3,
        description: '设备发生故障需要维修',
        requiredActions: ['故障诊断', '维修记录', '备件更换'],
        alertLevel: 'critical'
    },
    standby: {
        name: '待机',
        color: 'info',
        code: 4,
        description: '设备处于待机状态',
        requiredActions: ['状态检查'],
        alertLevel: 'info'
    }
}; 