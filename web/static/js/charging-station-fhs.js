/**
 * 充电桩故障处理系统
 */
class FaultHandlingSystem {
    constructor() {
        // 故障处理队列
        this.handlingQueue = new Map();
        // 处理历史记录
        this.handlingHistory = new Map();
        // 维修人员管理
        this.maintainers = new Map();
        // 备件管理
        this.spareParts = new Map();
        // 初始化
        this.initialize();
    }

    initialize() {
        this.initializeMaintainers();
        this.initializeSpareParts();
        this.startQueueProcessor();
    }

    /**
     * 初始化维修人员信息
     */
    initializeMaintainers() {
        // 添加模拟维修人员数据
        this.maintainers.set('M001', {
            id: 'M001',
            name: '张工',
            level: 'expert',
            skills: ['power', 'communication', 'temperature'],
            availability: true,
            contact: {
                phone: '13800138000',
                email: 'zhang@example.com'
            },
            currentTask: null
        });

        this.maintainers.set('M002', {
            id: 'M002',
            name: '李工',
            level: 'intermediate',
            skills: ['power', 'temperature'],
            availability: true,
            contact: {
                phone: '13800138001',
                email: 'li@example.com'
            },
            currentTask: null
        });

        this.maintainers.set('M003', {
            id: 'M003',
            name: '王工',
            level: 'basic',
            skills: ['temperature', 'basic-repair'],
            availability: true,
            contact: {
                phone: '13800138002',
                email: 'wang@example.com'
            },
            currentTask: null
        });
    }

    /**
     * 初始化备件信息
     */
    initializeSpareParts() {
        // 电源系统备件
        this.spareParts.set('power', new Map([
            ['PSU001', { name: '开关电源模块', stock: 5, threshold: 2 }],
            ['PCB001', { name: '主控制板', stock: 3, threshold: 1 }],
            ['CAP001', { name: '滤波电容', stock: 20, threshold: 5 }]
        ]));

        // 通信系统备件
        this.spareParts.set('communication', new Map([
            ['COM001', { name: '通信模块', stock: 4, threshold: 2 }],
            ['ANT001', { name: '天线组件', stock: 6, threshold: 2 }],
            ['CAB001', { name: '通信线缆', stock: 10, threshold: 3 }]
        ]));

        // 温控系统备件
        this.spareParts.set('temperature', new Map([
            ['FAN001', { name: '散热风扇', stock: 8, threshold: 3 }],
            ['SEN001', { name: '温度传感器', stock: 10, threshold: 4 }],
            ['HSK001', { name: '散热器组件', stock: 4, threshold: 2 }]
        ]));
    }

    /**
     * 启动队列处理器
     */
    startQueueProcessor() {
        setInterval(() => {
            this.processHandlingQueue();
        }, 5000); // 每5秒处理一次队列
    }

    /**
     * 处理故障
     * @param {string} stationId 充电站ID
     * @param {string} portId 充电端口ID
     * @param {Object} diagnosisReport 诊断报告
     */
    async handleFault(stationId, portId, diagnosisReport) {
        try {
            // 1. 创建处理任务
            const task = this.createHandlingTask(stationId, portId, diagnosisReport);

            // 2. 添加到处理队列
            this.addToHandlingQueue(task);

            // 3. 分配维修人员
            const assignedMaintainer = await this.assignMaintainer(task);
            task.maintainer = assignedMaintainer;

            // 4. 检查备件库存
            const requiredParts = this.checkRequiredParts(task);
            task.requiredParts = requiredParts;

            // 5. 生成处理计划
            const handlingPlan = this.generateHandlingPlan(task);
            task.handlingPlan = handlingPlan;

            // 6. 更新任务状态
            task.status = 'ready';
            this.updateTask(task);

            // 7. 通知相关人员
            await this.notifyRelatedParties(task);

            return task;
        } catch (error) {
            console.error('创建故障处理任务失败:', error);
            throw error;
        }
    }

    /**
     * 创建处理任务
     * @private
     */
    createHandlingTask(stationId, portId, diagnosisReport) {
        return {
            id: `T${Date.now()}`,
            stationId,
            portId,
            diagnosisReport,
            createTime: new Date(),
            priority: this.calculateTaskPriority(diagnosisReport),
            status: 'created',
            maintainer: null,
            requiredParts: [],
            handlingPlan: null,
            progress: 0,
            logs: []
        };
    }

    /**
     * 计算任务优先级
     * @private
     */
    calculateTaskPriority(diagnosisReport) {
        let priority = 0;

        // 根据故障严重程度调整优先级
        switch (diagnosisReport.overallSeverity) {
            case 'critical':
                priority += 100;
                break;
            case 'warning':
                priority += 50;
                break;
            case 'notice':
                priority += 10;
                break;
        }

        // 根据故障类型调整优先级
        diagnosisReport.faults.forEach(fault => {
            switch (fault.category) {
                case 'power':
                    priority += 30;
                    break;
                case 'temperature':
                    priority += 20;
                    break;
                case 'communication':
                    priority += 10;
                    break;
            }
        });

        // 根据可信度调整优先级
        priority *= diagnosisReport.confidence;

        return Math.round(priority);
    }

    /**
     * 添加到处理队列
     * @private
     */
    addToHandlingQueue(task) {
        const key = `${task.stationId}-${task.portId}`;
        if (!this.handlingQueue.has(key)) {
            this.handlingQueue.set(key, []);
        }
        this.handlingQueue.get(key).push(task);

        // 按优先级排序
        this.handlingQueue.get(key).sort((a, b) => b.priority - a.priority);
    }

    /**
     * 分配维修人员
     * @private
     */
    async assignMaintainer(task) {
        // 获取所有可用的维修人员
        const availableMaintainers = Array.from(this.maintainers.values())
            .filter(m => m.availability);

        if (availableMaintainers.length === 0) {
            throw new Error('没有可用的维修人员');
        }

        // 根据故障类型和严重程度选择合适的维修人员
        const requiredSkills = new Set();
        task.diagnosisReport.faults.forEach(fault => {
            requiredSkills.add(fault.category);
        });

        // 按技能匹配度和级别排序
        const rankedMaintainers = availableMaintainers
            .map(maintainer => ({
                maintainer,
                score: this.calculateMaintainerScore(maintainer, requiredSkills, task)
            }))
            .sort((a, b) => b.score - a.score);

        // 选择得分最高的维修人员
        const selected = rankedMaintainers[0].maintainer;

        // 更新维修人员状态
        selected.availability = false;
        selected.currentTask = task.id;

        return selected;
    }

    /**
     * 计算维修人员评分
     * @private
     */
    calculateMaintainerScore(maintainer, requiredSkills, task) {
        let score = 0;

        // 技能匹配度
        const skillMatchCount = maintainer.skills
            .filter(skill => requiredSkills.has(skill)).length;
        score += (skillMatchCount / requiredSkills.size) * 50;

        // 级别权重
        switch (maintainer.level) {
            case 'expert':
                score += 30;
                break;
            case 'intermediate':
                score += 20;
                break;
            case 'basic':
                score += 10;
                break;
        }

        // 如果是紧急任务且维修人员是专家，额外加分
        if (task.diagnosisReport.overallSeverity === 'critical' && maintainer.level === 'expert') {
            score += 20;
        }

        return score;
    }

    /**
     * 检查所需备件
     * @private
     */
    checkRequiredParts(task) {
        const requiredParts = [];

        task.diagnosisReport.faults.forEach(fault => {
            const categoryParts = this.spareParts.get(fault.category);
            if (categoryParts) {
                // 根据故障类型确定所需备件
                switch (fault.category) {
                    case 'power':
                        if (fault.name.includes('过压') || fault.name.includes('欠压')) {
                            requiredParts.push({
                                id: 'PSU001',
                                name: '开关电源模块',
                                quantity: 1
                            });
                        }
                        break;
                    case 'communication':
                        if (fault.name.includes('通信中断')) {
                            requiredParts.push({
                                id: 'COM001',
                                name: '通信模块',
                                quantity: 1
                            });
                        }
                        break;
                    case 'temperature':
                        if (fault.name.includes('过温')) {
                            requiredParts.push({
                                id: 'FAN001',
                                name: '散热风扇',
                                quantity: 1
                            });
                        }
                        break;
                }
            }
        });

        // 检查库存
        requiredParts.forEach(part => {
            const stock = this.getPartStock(part.id);
            part.available = stock >= part.quantity;
            part.stock = stock;
        });

        return requiredParts;
    }

    /**
     * 获取备件库存
     * @private
     */
    getPartStock(partId) {
        for (const [category, parts] of this.spareParts) {
            if (parts.has(partId)) {
                return parts.get(partId).stock;
            }
        }
        return 0;
    }

    /**
     * 生成处理计划
     * @private
     */
    generateHandlingPlan(task) {
        const plan = {
            steps: [],
            estimatedTime: 0,
            requiredTools: new Set(),
            safetyPrecautions: []
        };

        // 1. 安全预防措施
        plan.safetyPrecautions = this.generateSafetyPrecautions(task);

        // 2. 处理步骤
        task.diagnosisReport.faults.forEach(fault => {
            const steps = this.generateStepsForFault(fault);
            plan.steps.push(...steps);
            
            // 累加预计时间
            plan.estimatedTime += steps.reduce((total, step) => total + (step.estimatedTime || 0), 0);

            // 收集所需工具
            steps.forEach(step => {
                if (step.tools) {
                    step.tools.forEach(tool => plan.requiredTools.add(tool));
                }
            });
        });

        // 3. 排序步骤
        plan.steps.sort((a, b) => a.sequence - b.sequence);

        return plan;
    }

    /**
     * 生成安全预防措施
     * @private
     */
    generateSafetyPrecautions(task) {
        const precautions = [
            {
                id: 'S001',
                description: '确保维修前断开电源',
                type: 'critical'
            },
            {
                id: 'S002',
                description: '佩戴必要的防护装备',
                type: 'required'
            }
        ];

        // 根据故障类型添加特定的安全措施
        task.diagnosisReport.faults.forEach(fault => {
            switch (fault.category) {
                case 'power':
                    precautions.push({
                        id: 'S003',
                        description: '使用绝缘工具和测试设备',
                        type: 'critical'
                    });
                    break;
                case 'temperature':
                    precautions.push({
                        id: 'S004',
                        description: '等待设备冷却后进行操作',
                        type: 'warning'
                    });
                    break;
            }
        });

        return precautions;
    }

    /**
     * 为故障生成处理步骤
     * @private
     */
    generateStepsForFault(fault) {
        const steps = [];
        let sequence = 1;

        // 1. 准备步骤
        steps.push({
            sequence: sequence++,
            type: 'preparation',
            description: '检查维修工具和备件',
            estimatedTime: 5,
            tools: ['工具箱']
        });

        // 2. 检查步骤
        fault.suggestedActions.forEach(action => {
            steps.push({
                sequence: sequence++,
                type: 'check',
                description: action,
                estimatedTime: 10,
                tools: this.getRequiredTools(action)
            });
        });

        // 3. 维修步骤
        const repairSteps = this.generateRepairSteps(fault);
        repairSteps.forEach(step => {
            step.sequence = sequence++;
            steps.push(step);
        });

        // 4. 测试步骤
        steps.push({
            sequence: sequence++,
            type: 'test',
            description: '功能测试和验证',
            estimatedTime: 15,
            tools: ['测试仪']
        });

        return steps;
    }

    /**
     * 生成维修步骤
     * @private
     */
    generateRepairSteps(fault) {
        const steps = [];

        switch (fault.category) {
            case 'power':
                steps.push(
                    {
                        type: 'repair',
                        description: '更换电源模块',
                        estimatedTime: 30,
                        tools: ['螺丝刀套装', '万用表'],
                        requiredParts: ['PSU001']
                    }
                );
                break;
            case 'communication':
                steps.push(
                    {
                        type: 'repair',
                        description: '更新通信模块固件',
                        estimatedTime: 20,
                        tools: ['编程器', '笔记本电脑'],
                        requiredParts: []
                    }
                );
                break;
            case 'temperature':
                steps.push(
                    {
                        type: 'repair',
                        description: '清理散热系统',
                        estimatedTime: 25,
                        tools: ['清洁工具', '散热膏'],
                        requiredParts: ['FAN001']
                    }
                );
                break;
        }

        return steps;
    }

    /**
     * 获取所需工具
     * @private
     */
    getRequiredTools(action) {
        const toolMap = {
            '检查电压': ['万用表'],
            '检查电流': ['钳形表'],
            '测试通信': ['网络分析仪'],
            '检查温度': ['红外测温仪'],
            '更换组件': ['螺丝刀套装', '万用表'],
            '清理散热': ['清洁工具套装']
        };

        for (const [keyword, tools] of Object.entries(toolMap)) {
            if (action.includes(keyword)) {
                return tools;
            }
        }

        return ['基础工具'];
    }

    /**
     * 处理队列处理器
     * @private
     */
    async processHandlingQueue() {
        for (const [key, tasks] of this.handlingQueue) {
            // 获取队列中第一个未开始的任务
            const pendingTask = tasks.find(t => t.status === 'ready');
            if (pendingTask) {
                try {
                    // 开始处理任务
                    await this.startTaskHandling(pendingTask);
                } catch (error) {
                    console.error(`处理任务 ${pendingTask.id} 失败:`, error);
                    // 记录错误并继续处理下一个任务
                    this.logTaskError(pendingTask, error);
                }
            }
        }
    }

    /**
     * 开始任务处理
     * @private
     */
    async startTaskHandling(task) {
        // 更新任务状态
        task.status = 'in_progress';
        task.startTime = new Date();
        this.updateTask(task);

        // 执行处理步骤
        for (const step of task.handlingPlan.steps) {
            try {
                // 执行步骤
                await this.executeStep(task, step);
                
                // 更新进度
                task.progress = (task.handlingPlan.steps.indexOf(step) + 1) / 
                               task.handlingPlan.steps.length * 100;
                this.updateTask(task);

            } catch (error) {
                throw new Error(`执行步骤 ${step.sequence} 失败: ${error.message}`);
            }
        }

        // 完成任务
        await this.completeTask(task);
    }

    /**
     * 执行处理步骤
     * @private
     */
    async executeStep(task, step) {
        // 记录步骤开始
        this.logTaskProgress(task, `开始执行步骤: ${step.description}`);

        // 模拟步骤执行
        await new Promise(resolve => setTimeout(resolve, step.estimatedTime * 1000));

        // 记录步骤完成
        this.logTaskProgress(task, `完成步骤: ${step.description}`);
    }

    /**
     * 完成任务
     * @private
     */
    async completeTask(task) {
        // 更新任务状态
        task.status = 'completed';
        task.completeTime = new Date();
        task.progress = 100;

        // 释放维修人员
        if (task.maintainer) {
            const maintainer = this.maintainers.get(task.maintainer.id);
            if (maintainer) {
                maintainer.availability = true;
                maintainer.currentTask = null;
            }
        }

        // 更新备件库存
        task.requiredParts.forEach(part => {
            this.updatePartStock(part.id, -part.quantity);
        });

        // 记录任务完成
        this.logTaskProgress(task, '任务完成');

        // 从处理队列中移除
        const key = `${task.stationId}-${task.portId}`;
        const queue = this.handlingQueue.get(key);
        if (queue) {
            const index = queue.findIndex(t => t.id === task.id);
            if (index !== -1) {
                queue.splice(index, 1);
            }
        }

        // 添加到历史记录
        this.addToHistory(task);

        // 通知相关人员
        await this.notifyTaskCompletion(task);
    }

    /**
     * 更新备件库存
     * @private
     */
    updatePartStock(partId, change) {
        for (const [category, parts] of this.spareParts) {
            if (parts.has(partId)) {
                const part = parts.get(partId);
                part.stock += change;

                // 检查是否达到补货阈值
                if (part.stock <= part.threshold) {
                    this.notifyLowStock(partId, part);
                }

                break;
            }
        }
    }

    /**
     * 添加到历史记录
     * @private
     */
    addToHistory(task) {
        const key = `${task.stationId}-${task.portId}`;
        if (!this.handlingHistory.has(key)) {
            this.handlingHistory.set(key, []);
        }
        this.handlingHistory.get(key).push(task);
    }

    /**
     * 记录任务进度
     * @private
     */
    logTaskProgress(task, message) {
        task.logs.push({
            time: new Date(),
            type: 'progress',
            message
        });
    }

    /**
     * 记录任务错误
     * @private
     */
    logTaskError(task, error) {
        task.logs.push({
            time: new Date(),
            type: 'error',
            message: error.message
        });
    }

    /**
     * 通知相关人员
     * @private
     */
    async notifyRelatedParties(task) {
        // 通知维修人员
        if (task.maintainer) {
            await this.sendNotification(task.maintainer.contact, {
                type: 'task_assigned',
                task: {
                    id: task.id,
                    stationId: task.stationId,
                    portId: task.portId,
                    priority: task.priority
                }
            });
        }

        // 通知站点管理员
        await this.sendNotification({
            role: 'station_admin',
            stationId: task.stationId
        }, {
            type: 'task_created',
            task: {
                id: task.id,
                stationId: task.stationId,
                portId: task.portId,
                severity: task.diagnosisReport.overallSeverity
            }
        });
    }

    /**
     * 通知任务完成
     * @private
     */
    async notifyTaskCompletion(task) {
        const notification = {
            type: 'task_completed',
            task: {
                id: task.id,
                stationId: task.stationId,
                portId: task.portId,
                startTime: task.startTime,
                completeTime: task.completeTime,
                maintainer: task.maintainer.name
            }
        };

        // 通知站点管理员
        await this.sendNotification({
            role: 'station_admin',
            stationId: task.stationId
        }, notification);

        // 通知维修主管
        await this.sendNotification({
            role: 'maintenance_supervisor'
        }, notification);
    }

    /**
     * 通知库存不足
     * @private
     */
    async notifyLowStock(partId, part) {
        await this.sendNotification({
            role: 'inventory_manager'
        }, {
            type: 'low_stock',
            part: {
                id: partId,
                name: part.name,
                currentStock: part.stock,
                threshold: part.threshold
            }
        });
    }

    /**
     * 发送通知
     * @private
     */
    async sendNotification(recipient, content) {
        // 这里应该实现实际的通知发送逻辑
        console.log('发送通知:', {
            recipient,
            content
        });
    }
}

// 导出故障处理系统
window.FaultHandlingSystem = FaultHandlingSystem; 