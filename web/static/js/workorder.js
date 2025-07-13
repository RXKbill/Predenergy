document.addEventListener('DOMContentLoaded', function() {
    initializeWorkOrder();
    updateWorkOrderCount();
    initializeDeviceSelection();
    initializeFaultCategories();
});

function initializeWorkOrder() {
    // 初始化工单列表
    initializeWorkOrderList();
    
    // 初始化工单详情
    initializeWorkOrderDetail();
    
    // 初始化工单处理
    initializeWorkOrderProcess();
    
    // 初始化图表
    initializeCharts();
    
    // 初始化照片上传
    initializePhotoUpload();
    
    // 初始化离线存储
    initializeOfflineStorage();
}

// 初始化工单列表
function initializeWorkOrderList() {
    // 为所有工单添加ID属性
    document.querySelectorAll('.workorder-item').forEach(item => {
        const workOrderId = item.querySelector('.workorder-info [class*="mdi-identifier"]')?.parentElement.textContent.trim();
        if (workOrderId) {
            item.setAttribute('data-workorder-id', workOrderId);
        }
    });

    // 绑定工单点击事件
    document.querySelectorAll('.workorder-item').forEach(item => {
        const actionBtn = item.querySelector('.btn');
        if (actionBtn) {
            actionBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                const action = this.textContent.trim();
                
                switch (action) {
                    case '立即处理':
                    case '开始处理':
                        moveWorkOrderToProcessing(item);
                        showProcessModal();
                        break;
                    case '完成工单':
                        completeWorkOrder(item);
                        break;
                    case '查看详情':
                        showDetailModal();
                        break;
                }
            });
        }
    });
    
    // 绑定刷新按钮
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshWorkOrders);
    }
    
    // 绑定筛选按钮
    const filterBtn = document.getElementById('filterBtn');
    if (filterBtn) {
        filterBtn.addEventListener('click', showFilterOptions);
    }
}

// 初始化工单详情
function initializeWorkOrderDetail() {
    // 初始化历史数据图表
    const historyChart = document.getElementById('historyChart');
    if (historyChart) {
        new Chart(historyChart, {
            type: 'line',
            data: {
                labels: ['3-8', '3-9', '3-10', '3-11', '3-12', '3-13', '3-14'],
                datasets: [{
                    label: '振动频率',
                    data: [12, 19, 15, 17, 25, 28, 35],
                    borderColor: '#2196f3',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '近7天振动频率趋势'
                    }
                }
            }
        });
    }
    
    // 初始化预测趋势图表
    const predictionChart = document.getElementById('predictionChart');
    if (predictionChart) {
        new Chart(predictionChart, {
            type: 'line',
            data: {
                labels: ['现在', '2h后', '4h后', '6h后', '8h后', '10h后', '12h后'],
                datasets: [{
                    label: '预测振动频率',
                    data: [35, 38, 42, 45, 50, 54, 60],
                    borderColor: '#dc3545',
                    tension: 0.4,
                    borderDash: [5, 5]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '未来12小时预测趋势'
                    }
                }
            }
        });
    }
}

// 初始化工单处理
function initializeWorkOrderProcess() {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    
    if (prevBtn && nextBtn) {
        prevBtn.addEventListener('click', () => {
            const currentStep = getCurrentStep();
            if (currentStep > 1) {
                updateStep(currentStep - 1);
            }
        });
        
        nextBtn.addEventListener('click', () => {
            const currentStep = getCurrentStep();
            if (validateStep(currentStep)) {
                if (currentStep < 4) {
                    updateStep(currentStep + 1);
                } else {
                    submitWorkOrder();
                }
            }
        });
    }
}

// 移除旧的事件监听器
const oldNextBtn = document.getElementById('nextStep');
if (oldNextBtn) {
    const newNextBtn = oldNextBtn.cloneNode(true);
    oldNextBtn.parentNode.replaceChild(newNextBtn, oldNextBtn);
}

// 更新处理步骤
function updateStep(step) {
    // 更新步骤指示器
    document.querySelectorAll('.step').forEach((el, index) => {
        if (index + 1 < step) {
            el.classList.add('completed');
            el.classList.remove('active');
        } else if (index + 1 === step) {
            el.classList.add('active');
            el.classList.remove('completed');
        } else {
            el.classList.remove('active', 'completed');
        }
    });
    
    // 显示当前步骤内容
    document.querySelectorAll('.step-content').forEach((el, index) => {
        el.style.display = index + 1 === step ? 'block' : 'none';
    });
    
    // 更新按钮状态
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    
    if (prevBtn) {
        prevBtn.disabled = step === 1;
    }
    
    if (nextBtn) {
        nextBtn.textContent = step === 4 ? '提交' : '下一步';
    }
    
    // 特殊步骤处理
    if (step === 2) {
        initializeFaultCategories();
    } else if (step === 4) {
        loadTestData();
    }
}

// 获取当前步骤
function getCurrentStep() {
    const activeStep = document.querySelector('.step.active');
    return activeStep ? Array.from(document.querySelectorAll('.step')).indexOf(activeStep) + 1 : 1;
}

// 前往下一步
function goToNextStep() {
    const currentStep = getCurrentStep();
    updateStep(currentStep + 1);
}

// 验证步骤
function validateStep(step) {
    if (step === 3) {
        return validateStep3();
    } else if (step === 4) {
        return validateStep4();
    }

    const stepContent = document.getElementById(`step${step}`);
    if (!stepContent) return true;
    
    let isValid = true;
    const requiredFields = stepContent.querySelectorAll('input[required], textarea[required], select[required]');
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            isValid = false;
            field.classList.add('is-invalid');
            showToast(`请填写${field.closest('.mb-3').querySelector('.form-label').textContent}`, 'warning');
        } else {
            field.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// 验证第四步的表单（质检确认）
function validateStep4() {
    // 验证质检项目
    const qualityChecks = document.querySelectorAll('.quality-checks .form-check-input');
    const allChecked = Array.from(qualityChecks).every(check => check.checked);
    if (!allChecked) {
        showToast('请完成所有质检项目的检查', 'warning');
        return false;
    }

    // 验证质检结论
    const conclusion = document.querySelector('select[required]');
    if (!conclusion || !conclusion.value) {
        showToast('请选择质检结论', 'warning');
        return false;
    }

    // 验证质检意见
    const opinion = document.querySelector('textarea[required]');
    if (!opinion || !opinion.value.trim()) {
        showToast('请填写质检意见', 'warning');
        return false;
    }

    return true;
}

// 初始化照片上传
function initializePhotoUpload() {
    document.querySelectorAll('.photo-upload button').forEach(btn => {
        btn.addEventListener('click', () => {
            // 检查是否支持调用相机
            if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
                takePhoto();
            } else {
                // 回退到文件选择
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = 'image/*';
                input.onchange = handlePhotoSelect;
                input.click();
            }
        });
    });
}

// 拍照功能
async function takePhoto() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.createElement('video');
        video.srcObject = stream;
        
        // 创建拍照界面
        const modal = createPhotoModal(video, stream);
        document.body.appendChild(modal);
        
        video.play();
    } catch (err) {
        console.error('相机访问失败:', err);
        showToast('无法访问相机，请检查权限设置', 'error');
    }
}

// 创建拍照模态框
function createPhotoModal(video, stream) {
    const modal = document.createElement('div');
    modal.className = 'photo-modal';
    modal.innerHTML = `
        <div class="photo-preview">
            ${video.outerHTML}
            <div class="photo-controls">
                <button class="btn btn-light btn-capture">
                    <i class="mdi mdi-camera"></i>
                </button>
                <button class="btn btn-light btn-close">
                    <i class="mdi mdi-close"></i>
                </button>
            </div>
        </div>
    `;
    
    // 绑定拍照按钮
    modal.querySelector('.btn-capture').onclick = () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        // 获取照片数据
        const photoData = canvas.toDataURL('image/jpeg');
        addPhotoToPreview(photoData);
        
        // 关闭相机
        stream.getTracks().forEach(track => track.stop());
        modal.remove();
    };
    
    // 绑定关闭按钮
    modal.querySelector('.btn-close').onclick = () => {
        stream.getTracks().forEach(track => track.stop());
        modal.remove();
    };
    
    return modal;
}

// 处理照片选择
function handlePhotoSelect(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            addPhotoToPreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }
}

// 添加照片到预览区
function addPhotoToPreview(photoData) {
    const preview = document.querySelector('.photo-preview');
    const img = document.createElement('img');
    img.src = photoData;
    
    // 添加删除按钮
    const wrapper = document.createElement('div');
    wrapper.className = 'photo-item';
    wrapper.innerHTML = `
        <button class="btn btn-sm btn-danger btn-delete">
            <i class="mdi mdi-close"></i>
        </button>
    `;
    wrapper.prepend(img);
    
    // 绑定删除事件
    wrapper.querySelector('.btn-delete').onclick = () => wrapper.remove();
    
    preview.appendChild(wrapper);
}

// 初始化离线存储
function initializeOfflineStorage() {
    // 检查浏览器是否支持 IndexedDB
    if (!window.indexedDB) {
        console.log('浏览器不支持 IndexedDB');
        return;
    }
    
    // 打开数据库
    const request = indexedDB.open('workOrderDB', 1);
    
    request.onerror = function(event) {
        console.error('数据库打开失败:', event.target.error);
    };
    
    request.onupgradeneeded = function(event) {
        const db = event.target.result;
        
        // 创建工单对象仓库
        if (!db.objectStoreNames.contains('workOrders')) {
            const store = db.createObjectStore('workOrders', { keyPath: 'id' });
            store.createIndex('status', 'status');
            store.createIndex('timestamp', 'timestamp');
        }
        
        // 创建照片对象仓库
        if (!db.objectStoreNames.contains('photos')) {
            const store = db.createObjectStore('photos', { keyPath: 'id' });
            store.createIndex('workOrderId', 'workOrderId');
        }
    };
    
    request.onsuccess = function(event) {
        const db = event.target.result;
        window.workOrderDB = db;
        
        // 检查并同步离线数据
        syncOfflineData();
    };
}

// 同步离线数据
async function syncOfflineData() {
    if (!navigator.onLine) {
        console.log('设备离线，暂不同步');
        return;
    }
    
    const db = window.workOrderDB;
    if (!db) return;
    
    const transaction = db.transaction(['workOrders'], 'readonly');
    const store = transaction.objectStore('workOrders');
    
    // 获取所有未同步的工单
    const request = store.index('status').getAll('pending_sync');
    
    request.onsuccess = async function() {
        const pendingWorkOrders = request.result;
        
        for (const workOrder of pendingWorkOrders) {
            try {
                // 尝试同步到服务器
                await syncWorkOrder(workOrder);
                
                // 更新本地状态
                const updateTx = db.transaction(['workOrders'], 'readwrite');
                const updateStore = updateTx.objectStore('workOrders');
                workOrder.status = 'synced';
                updateStore.put(workOrder);
            } catch (err) {
                console.error('工单同步失败:', err);
            }
        }
    };
}

// 同步单个工单到服务器
async function syncWorkOrder(workOrder) {
    try {
        const response = await fetch('/api/workorders', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(workOrder)
        });
        
        if (!response.ok) {
            throw new Error('同步失败');
        }
        
        showToast('工单已同步', 'success');
    } catch (err) {
        showToast('工单同步失败，将在网络恢复后重试', 'warning');
        throw err;
    }
}

// 显示工单详情模态框
function showDetailModal() {
    const modal = new bootstrap.Modal(document.getElementById('workorderDetailModal'));
    modal.show();
}

// 显示工单处理模态框
function showProcessModal() {
    const modal = new bootstrap.Modal(document.getElementById('workorderProcessModal'));
    modal.show();
}

// 完成工单
function completeWorkOrder(workOrderItem) {
    // 获取工单ID
    const workOrderId = workOrderItem.getAttribute('data-workorder-id');
    
    // 显示确认模态框
    const modal = new bootstrap.Modal(document.getElementById('completeWorkOrderModal'));
    modal.show();
    
    // 绑定确认按钮事件
    const confirmBtn = document.getElementById('confirmComplete');
    const oldConfirmHandler = confirmBtn.onclick;
    
    confirmBtn.onclick = () => {
        try {
            // 移动工单到已完成列表
            moveWorkOrderToCompleted(workOrderId);
            
            // 显示成功提示
            showToast('工单已完成', 'success');
            
            // 更新工单数量标签
            updateWorkOrderCount();
            
            // 关闭模态框
            modal.hide();
        } catch (error) {
            console.error('完成工单失败:', error);
            showToast('操作失败，请重试', 'error');
        }
        
        // 移除事件处理器
        confirmBtn.onclick = oldConfirmHandler;
    };
}

// 刷新工单列表
function refreshWorkOrders() {
    // TODO: 实现工单刷新逻辑
    showToast('正在刷新...', 'info');
}

// 显示筛选选项
function showFilterOptions() {
    // TODO: 实现筛选功能
    showToast('筛选功能开发中', 'info');
}

// 显示提示消息
function showToast(message, type = 'info') {
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
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    const container = document.createElement('div');
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.appendChild(toast);
    document.body.appendChild(container);
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', () => {
        document.body.removeChild(container);
    });
}

// 从巡检发现生成工单
function createWorkOrder(title, priority) {
    const now = new Date();
    const workOrderId = `WO-${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(Math.floor(Math.random() * 1000)).padStart(3, '0')}`;
    
    // 创建工单HTML
    const workOrderHtml = `
        <div class="workorder-item ${priority}-priority">
            <div class="priority-badge">
                <i class="mdi mdi-alert${priority === 'medium' ? '-outline' : ''}"></i>
                ${priority === 'high' ? '紧急' : '普通'}
            </div>
            <div class="workorder-content">
                <h6 class="workorder-title">${title}</h6>
                <p class="workorder-desc">由巡检报告自动生成</p>
                <div class="workorder-info">
                    <span><i class="mdi mdi-clock-outline"></i> 刚刚</span>
                    <span><i class="mdi mdi-identifier"></i> ${workOrderId}</span>
                </div>
            </div>
            <button class="btn btn-primary btn-sm">立即处理</button>
        </div>
    `;
    
    // 添加到待处理工单列表
    const pendingList = document.querySelector('#pending .workorder-list');
    pendingList.insertAdjacentHTML('afterbegin', workOrderHtml);
    
    // 更新工单数量标签
    updateWorkOrderCount();
    
    // 显示提示
    showToast('工单已生成', 'success');
    
    // 绑定新工单的点击事件
    initializeWorkOrderList();
}

// 开始巡检
function startInspection() {
    // 获取选中的设备
    const selectedDevices = Array.from(document.querySelectorAll('.device-list-section .form-check-input:checked'))
        .map(checkbox => checkbox.value);
    
    if (selectedDevices.length === 0) {
        showToast('请选择需要巡检的设备', 'warning');
        return;
    }
    
    // 跳转到巡检页面
    window.location.href = 'app-drone_inspection_and_monitoring.html';
}

// 更新工单数量标签
function updateWorkOrderCount() {
    const pendingCount = document.querySelectorAll('#pending .workorder-item').length;
    const processingCount = document.querySelectorAll('#processing .workorder-item').length;
    const completedCount = document.querySelectorAll('#completed .workorder-item').length;
    
    // 更新标签数字
    const pendingBadge = document.querySelector('[data-bs-target="#pending"] .badge');
    const processingBadge = document.querySelector('[data-bs-target="#processing"] .badge');
    const completedBadge = document.querySelector('[data-bs-target="#completed"] .badge');
    
    if (pendingBadge) pendingBadge.textContent = pendingCount;
    if (processingBadge) processingBadge.textContent = processingCount;
    if (completedBadge) completedBadge.textContent = completedCount;
    
    // 如果所有列表都为空，显示空状态提示
    ['pending', 'processing', 'completed'].forEach(status => {
        const list = document.querySelector(`#${status} .workorder-list`);
        const emptyState = document.querySelector(`#${status} .empty-state`);
        
        if (list && list.children.length === 0) {
            if (!emptyState) {
                const emptyStateDiv = document.createElement('div');
                emptyStateDiv.className = 'empty-state text-center text-muted p-4';
                emptyStateDiv.innerHTML = `
                    <i class="mdi mdi-clipboard-text-outline display-4"></i>
                    <p class="mt-2">暂无${status === 'pending' ? '待处理' : status === 'processing' ? '处理中' : '已完成'}工单</p>
                `;
                list.parentNode.appendChild(emptyStateDiv);
            }
        } else if (emptyState) {
            emptyState.remove();
        }
    });
}

// 初始化设备选择功能
function initializeDeviceSelection() {
    const checkboxes = document.querySelectorAll('.device-list-section .form-check-input');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const selectedCount = document.querySelectorAll('.device-list-section .form-check-input:checked').length;
            const startButton = document.querySelector('.device-list-section .btn-primary');
            startButton.disabled = selectedCount === 0;
        });
    });
}

// 故障类别和子类别映射
const faultCategories = {
    blade: ['裂痕', '变形', '腐蚀', '磨损', '结冰'],
    generator: ['轴承故障', '绕组短路', '散热异常', '电压异常'],
    control: ['通信故障', '传感器异常', '控制器故障', '软件错误'],
    other: ['其他问题']
};

// 初始化故障类别联动
function initializeFaultCategories() {
    const categorySelect = document.getElementById('faultCategory');
    const subCategorySelect = document.getElementById('faultSubCategory');
    
    if (categorySelect && subCategorySelect) {
        categorySelect.addEventListener('change', function() {
            const category = this.value;
            const subFaults = faultCategories[category] || [];
            
            // 清空并重新填充子类别选项
            subCategorySelect.innerHTML = '<option value="">选择具体故障</option>';
            subFaults.forEach(fault => {
                const option = document.createElement('option');
                option.value = fault;
                option.textContent = fault;
                subCategorySelect.appendChild(option);
            });
        });
    }
}

// 查看巡检详情
function viewInspectionDetail() {
    // 跳转到巡检详情页面
    window.location.href = 'app-drone_inspection_and_monitoring.html';
}

// 拍照功能增强
function takePhoto(mode) {
    if (mode === 'thermal') {
        // 调用热成像相机
        showToast('正在启动热成像相机...', 'info');
        // TODO: 实现热成像拍摄
    } else {
        // 调用普通相机
        if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
            startCamera();
        } else {
            fallbackToFileUpload();
        }
    }
}

// 添加配件
function addPart() {
    const partNo = document.getElementById('partNo').value;
    const partName = document.getElementById('partName').value;
    const quantity = document.getElementById('partQuantity').value;
    
    if (!partNo || !partName || !quantity) {
        showToast('请填写完整的配件信息', 'warning');
        return;
    }
    
    const partsList = document.querySelector('.parts-list');
    const partItem = document.createElement('div');
    partItem.className = 'part-item d-flex justify-content-between align-items-center p-2 border-bottom';
    partItem.innerHTML = `
        <div>
            <small class="text-muted">${partNo}</small>
            <div>${partName} × ${quantity}</div>
        </div>
        <button type="button" class="btn btn-sm btn-outline-danger" onclick="this.parentElement.remove()">
            <i class="mdi mdi-delete"></i>
        </button>
    `;
    
    partsList.appendChild(partItem);
    
    // 清空输入
    document.getElementById('partNo').value = '';
    document.getElementById('partName').value = '';
    document.getElementById('partQuantity').value = '';
}

// 添加维修步骤
function addRepairStep() {
    const stepInput = document.getElementById('repairStep');
    const stepText = stepInput.value.trim();
    
    if (!stepText) {
        showToast('请输入维修步骤', 'warning');
        return;
    }
    
    const stepsList = document.querySelector('.repair-steps-list');
    const stepItem = document.createElement('div');
    stepItem.className = 'step-item d-flex justify-content-between align-items-center p-2 border-bottom';
    stepItem.innerHTML = `
        <div class="d-flex align-items-center">
            <span class="step-number me-2">${stepsList.children.length + 1}.</span>
            <span>${stepText}</span>
        </div>
        <button type="button" class="btn btn-sm btn-outline-danger" onclick="removeRepairStep(this)">
            <i class="mdi mdi-delete"></i>
        </button>
    `;
    
    stepsList.appendChild(stepItem);
    stepInput.value = '';
}

// 删除维修步骤
function removeRepairStep(button) {
    const stepsList = document.querySelector('.repair-steps-list');
    button.parentElement.remove();
    
    // 重新编号
    Array.from(stepsList.children).forEach((step, index) => {
        step.querySelector('.step-number').textContent = `${index + 1}.`;
    });
}

// 加载测试数据
function loadTestData() {
    // TODO: 从设备获取实时数据
    // 这里使用模拟数据
    const testData = {
        noLoadSpeed: 1500,
        loadPower: 2.2,
        vibration: 0.15,
        temperature: 25
    };
    
    // 更新显示
    updateTestDataDisplay(testData);
}

// 更新测试数据显示
function updateTestDataDisplay(data) {
    const testDataContainer = document.querySelector('.test-data');
    if (!testDataContainer) return;
    
    // 更新各项数据的显示
    Object.entries(data).forEach(([key, value]) => {
        const element = testDataContainer.querySelector(`[data-test="${key}"]`);
        if (element) {
            element.textContent = value;
            
            // 更新状态标签
            const badge = element.nextElementSibling;
            if (badge) {
                const isNormal = checkValueInRange(key, value);
                badge.className = `badge ${isNormal ? 'bg-success' : 'bg-danger'}`;
                badge.textContent = isNormal ? '正常' : '异常';
            }
        }
    });
}

// 检查数值是否在正常范围内
function checkValueInRange(key, value) {
    const ranges = {
        noLoadSpeed: [1400, 1600],
        loadPower: [2.0, 2.4],
        vibration: [0, 0.2],
        temperature: [0, 30]
    };
    
    const range = ranges[key];
    return range ? value >= range[0] && value <= range[1] : true;
}

// 提交工单
async function submitWorkOrder() {
    const currentStep = getCurrentStep();
    if (currentStep === 4 && !validateStep4()) {
        return;
    }
    
    try {
        // 显示加载状态
        showLoadingState();
        
        // 收集所有数据
        const formData = collectFormData();
        
        // 上传图片
        const imageUrls = await uploadImages();
        formData.images = imageUrls;
        
        // 提交工单数据
        const response = await submitWorkOrderData(formData);
        
        if (response.success) {
            // 更新工单状态
            moveWorkOrderToCompleted(formData.workOrderId);
            
            showToast('工单已完成', 'success');
            
            // 关闭处理模态框
            const processModal = bootstrap.Modal.getInstance(document.getElementById('workorderProcessModal'));
            if (processModal) {
                processModal.hide();
            }
            
            // 更新工单数量标签
            updateWorkOrderCount();
        } else {
            throw new Error(response.message);
        }
    } catch (error) {
        showToast(error.message || '提交失败，请重试', 'error');
    } finally {
        hideLoadingState();
    }
}

// 收集表单数据
function collectFormData() {
    // TODO: 实现表单数据收集
    return {
        workOrderId: document.getElementById('workOrderId').textContent,
        deviceId: document.getElementById('deviceId').textContent,
        // ... 其他数据
    };
}

// 上传图片
async function uploadImages() {
    // TODO: 实现图片上传
    return [];
}

// 提交工单数据
async function submitWorkOrderData(data) {
    // TODO: 实现数据提交
    return { success: true };
}

// 显示加载状态
function showLoadingState() {
    const submitBtn = document.getElementById('nextStep');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>提交中...';
    }
}

// 隐藏加载状态
function hideLoadingState() {
    const submitBtn = document.getElementById('nextStep');
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = '提交';
    }
}

// 验证所有步骤
function validateAllSteps() {
    let isValid = true;
    const form = document.getElementById('workorderForm');
    
    // 验证必填字段
    const requiredFields = form.querySelectorAll('[required]');
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            isValid = false;
            field.classList.add('is-invalid');
        } else {
            field.classList.remove('is-invalid');
        }
    });
    
    // 验证照片上传
    const photoPreview = document.querySelector('.photo-preview');
    if (photoPreview && !photoPreview.children.length) {
        isValid = false;
        showToast('请上传必要的照片', 'warning');
    }
    
    // 验证质检项目
    const qualityChecks = document.querySelectorAll('.quality-checks .form-check-input');
    const allChecked = Array.from(qualityChecks).every(check => check.checked);
    if (!allChecked) {
        isValid = false;
        showToast('请完成所有质检项目', 'warning');
    }
    
    return isValid;
}

// 验证第三步的表单
function validateStep3() {
    // 获取必填字段
    const repairPlan = document.getElementById('repairPlan').value;
    const repairDesc = document.getElementById('repairDesc').value;
    const testSpeed = document.getElementById('testSpeed').value;
    const testPower = document.getElementById('testPower').value;
    const hasPhoto = document.getElementById('hasPhoto').value;
    
    // 检查是否有至少一个维修步骤
    const hasRepairSteps = document.querySelector('.repair-steps-list').children.length > 0;
    
    if (!repairPlan || !repairDesc || !testSpeed || !testPower) {
        showToast('请填写所有必填项', 'warning');
        return false;
    }
    
    if (!hasRepairSteps) {
        showToast('请至少添加一个维修步骤', 'warning');
        return false;
    }
    
    if (!hasPhoto) {
        showToast('请上传或拍摄维修后照片', 'warning');
        return false;
    }
    
    return true;
}

// 模拟拍照上传
function simulatePhotoUpload() {
    // 在演示环境中，直接显示一个成功消息
    showToast('拍照上传成功', 'success');
    document.getElementById('hasPhoto').value = '1';
}

// 处理文件上传
function handlePhotoUpload(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const photoPreview = document.querySelector('.photo-preview');
            const photoItem = document.createElement('div');
            photoItem.className = 'photo-item';
            photoItem.innerHTML = `
                <img src="${e.target.result}" class="img-fluid rounded" alt="维修后照片">
                <button class="btn btn-sm btn-danger btn-delete" onclick="removePhoto(this)">
                    <i class="mdi mdi-close"></i>
                </button>
            `;
            photoPreview.appendChild(photoItem);
            document.getElementById('hasPhoto').value = '1';
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// 删除照片
function removePhoto(button) {
    button.closest('.photo-item').remove();
    const photoPreview = document.querySelector('.photo-preview');
    if (photoPreview.children.length === 0) {
        document.getElementById('hasPhoto').value = '';
    }
}

// 将工单移动到处理中状态
function moveWorkOrderToProcessing(workOrderItem) {
    const workOrderId = workOrderItem.getAttribute('data-workorder-id');
    const processingList = document.querySelector('#processing .workorder-list');
    
    // 创建处理中状态的工单
    const processingWorkOrder = document.createElement('div');
    processingWorkOrder.className = 'workorder-item';
    processingWorkOrder.setAttribute('data-workorder-id', workOrderId);
    processingWorkOrder.innerHTML = `
        <div class="priority-badge">
            <i class="mdi mdi-progress-clock"></i>
            处理中
        </div>
        <div class="workorder-content">
            <h6 class="workorder-title">${workOrderItem.querySelector('.workorder-title').textContent}</h6>
            <p class="workorder-desc">${workOrderItem.querySelector('.workorder-desc').textContent}</p>
            <div class="workorder-info">
                <span><i class="mdi mdi-clock-outline"></i> 进行中</span>
                <span><i class="mdi mdi-map-marker"></i> ${workOrderItem.querySelector('.workorder-info [class*="mdi-map-marker"]').parentElement.textContent.trim()}</span>
                <span><i class="mdi mdi-identifier"></i> ${workOrderId}</span>
            </div>
            <div class="progress mt-2" style="height: 6px;">
                <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
            </div>
        </div>
        <button class="btn btn-success btn-sm" onclick="completeWorkOrder(this.closest('.workorder-item'))">完成工单</button>
    `;

    // 添加到处理中列表
    processingList.insertBefore(processingWorkOrder, processingList.firstChild);
    
    // 从待处理列表中移除
    workOrderItem.remove();
    
    // 更新工单数量
    updateWorkOrderCount();
}

// 将工单移动到已完成列表
function moveWorkOrderToCompleted(workOrderId) {
    // 查找当前工单元素
    const workOrderItem = document.querySelector(`[data-workorder-id="${workOrderId}"]`) || 
                         document.querySelector(`.workorder-item:has(.workorder-info:contains("${workOrderId}"))`);
    
    if (!workOrderItem) {
        console.error('未找到工单元素:', workOrderId);
        return;
    }
    
    // 创建已完成状态的工单HTML
    const completedWorkOrder = document.createElement('div');
    completedWorkOrder.className = 'workorder-item completed';
    completedWorkOrder.innerHTML = `
        <div class="priority-badge">
            <i class="mdi mdi-check-circle"></i>
            已完成
        </div>
        <div class="workorder-content">
            <h6 class="workorder-title">${workOrderItem.querySelector('.workorder-title').textContent}</h6>
            <p class="workorder-desc">${workOrderItem.querySelector('.workorder-desc').textContent}</p>
            <div class="workorder-info">
                <span><i class="mdi mdi-clock-outline"></i> 完成于 ${new Date().toLocaleDateString()}</span>
                <span><i class="mdi mdi-map-marker"></i> ${workOrderItem.querySelector('.workorder-info [class*="mdi-map-marker"]').parentElement.textContent.trim()}</span>
                <span><i class="mdi mdi-identifier"></i> ${workOrderId}</span>
            </div>
        </div>
        <button class="btn btn-outline-secondary btn-sm" onclick="showDetailModal()">查看详情</button>
    `;

    // 将工单添加到已完成列表
    const completedList = document.querySelector('#completed .workorder-list');
    completedList.insertBefore(completedWorkOrder, completedList.firstChild);

    // 从原列表中移除工单
    workOrderItem.remove();

    // 更新工单数量标签
    updateWorkOrderCount();
} 