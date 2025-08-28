document.addEventListener('DOMContentLoaded', function() {
    let currentPasaran = document.getElementById('pasaranSelect').value;
    let trainingInterval = null;
    let evaluationInterval = null;
    let updateInterval = null; 
    const charts = {};

    const ui = {
        pasaranSelect: document.getElementById('pasaranSelect'),
        trainingMode: document.getElementById('trainingMode'),
        startTrainingBtn: document.getElementById('startTrainingBtn'),
        updateDataBtn: document.getElementById('updateDataBtn'),
        // UPDATED: Menambahkan checkbox ke UI object
        recencyBiasCheck: document.getElementById('recencyBiasCheck'),
        dataInfo: document.getElementById('dataInfo'),
        predictBtn: document.getElementById('predictBtn'),
        predictionDateInput: document.getElementById('predictionDate'),
        predictionDisplay: document.getElementById('prediction-display'),
        predDate: document.getElementById('predDate'),
        predictionResult: document.getElementById('predictionResult'),
        kandidatAs: document.getElementById('kandidatAs'),
        kandidatKop: document.getElementById('kandidatKop'),
        kandidatKepala: document.getElementById('kandidatKepala'),
        kandidatEkor: document.getElementById('kandidatEkor'),
        angkaMain: document.getElementById('angkaMain'),
        colokBebas: document.getElementById('colokBebas'),
        modelStatus: document.getElementById('modelStatus'),
        systemActivityStatus: document.getElementById('systemActivityStatus'),
        trainingProgress: document.querySelector('.progress'),
        trainingProgressBar: document.getElementById('trainingProgressBar'),
        healthTab: document.getElementById('health-tab'),
        evaluationTab: document.getElementById('evaluation-tab'),
        evalStartDate: document.getElementById('evalStartDate'),
        evalEndDate: document.getElementById('evalEndDate'),
        startEvaluationBtn: document.getElementById('startEvaluationBtn'),
        evaluationResultArea: document.getElementById('evaluationResultArea'),
        evaluationStatus: document.getElementById('evaluationStatus'),
        activePasaranForEval: document.getElementById('activePasaranForEval'),
        evaluationSummaryCards: document.getElementById('evaluationSummaryCards'),
        evalTotalDays: document.getElementById('evalTotalDays'),
        evalAsAccuracy: document.getElementById('evalAsAccuracy'),
        evalKopAccuracy: document.getElementById('evalKopAccuracy'),
        evalKepalaAccuracy: document.getElementById('evalKepalaAccuracy'),
        evalEkorAccuracy: document.getElementById('evalEkorAccuracy'),
        evalAmAccuracy: document.getElementById('evalAmAccuracy'),
        evalCbAccuracy: document.getElementById('evalCbAccuracy'),
        retrainingRecommendation: document.getElementById('retrainingRecommendation'),
        evaluationDetailTableBody: document.getElementById('evaluationDetailTableBody'),
        driftLog: document.getElementById('driftLog'),
        refreshFeatureImportance: document.getElementById('refreshFeatureImportance'),
        refreshDriftLog: document.getElementById('refreshDriftLog'),
        activePasaranForHealth: document.getElementById('activePasaranForHealth')
    };

    function showAlert(message, type = 'info') {
        const alertWrapper = document.createElement('div');
        alertWrapper.className = 'toast-container position-fixed top-0 end-0 p-3';
        alertWrapper.style.zIndex = 1055;
        alertWrapper.innerHTML = `<div class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true"><div class="d-flex"><div class="toast-body">${message}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button></div></div>`;
        document.body.appendChild(alertWrapper);
        const toast = new bootstrap.Toast(alertWrapper.querySelector('.toast'));
        toast.show();
        alertWrapper.addEventListener('hidden.bs.toast', () => alertWrapper.remove());
    }

    async function fetchData(url, options = {}) {
        try {
            const response = await fetch(url, options);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || data.details || `HTTP error! status: ${response.status}`);
            }
            return data;
        } catch (error) {
            console.error('Fetch error:', error);
            showAlert(error.message, 'danger');
            return null;
        }
    }

    function updateActivePasaranDisplay() {
        const pasaranDisplayName = ui.pasaranSelect.options[ui.pasaranSelect.selectedIndex].text;
        ui.activePasaranForEval.textContent = pasaranDisplayName;
        ui.activePasaranForHealth.textContent = pasaranDisplayName;
    }

    function updateSystemStatus() {
        if (!currentPasaran) return;
        fetchData(`/debug/model-status/${currentPasaran}`).then(data => {
            if (!data) return;
            ui.modelStatus.innerHTML = data.models_ready ? 
                `<span class="badge bg-success">Siap</span>` : 
                `<span class="badge bg-danger">Perlu Training</span>`;
            ui.dataInfo.innerHTML = data.data_manager_df_shape ? 
                `<i class="fas fa-database me-2"></i>${data.data_manager_df_shape[0]} baris data.` : 
                `<i class="fas fa-exclamation-triangle me-2"></i>Data tidak ditemukan.`;
        });
    }

    function getPrediction() {
        const btn = ui.predictBtn;
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Memprediksi...`;
        const formData = new FormData();
        formData.append('pasaran', currentPasaran);
        formData.append('prediction_date', ui.predictionDateInput.value);
        fetchData('/predict', { method: 'POST', body: formData }).then(data => {
            if (data && data.final_4d_prediction) {
                ui.predDate.textContent = data.prediction_date;
                ui.predictionResult.textContent = data.final_4d_prediction;
                ui.kandidatAs.textContent = data.kandidat_as;
                ui.kandidatKop.textContent = data.kandidat_kop;
                ui.kandidatKepala.textContent = data.kandidat_kepala;
                ui.kandidatEkor.textContent = data.kandidat_ekor;
                ui.angkaMain.textContent = data.angka_main;
                ui.colokBebas.textContent = data.colok_bebas;
                ui.predictionDisplay.style.display = 'block';
                showAlert('Prediksi berhasil dibuat dengan data terbaru.', 'success');
            } else {
                ui.predictionDisplay.style.display = 'none';
                showAlert('Tidak dapat membuat prediksi', 'warning');
            }
        }).finally(() => {
            btn.disabled = false;
            btn.innerHTML = originalText;
        });
    }
    
    function startUpdateData() {
        const btn = ui.updateDataBtn;
        btn.dataset.originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Mengupdate...`;
        ui.systemActivityStatus.textContent = 'Memulai sinkronisasi data...';
        const formData = new FormData();
        formData.append('pasaran', currentPasaran);
        fetchData('/update-data', { method: 'POST', body: formData }).then(data => {
            if (data?.status === 'success') {
                showAlert(data.message, 'info');
                monitorUpdateStatus();
            } else {
                btn.disabled = false;
                btn.innerHTML = btn.dataset.originalText;
                ui.systemActivityStatus.textContent = 'Gagal memulai update data.';
            }
        });
    }

    function monitorUpdateStatus() {
        if (updateInterval) clearInterval(updateInterval);
        ui.trainingProgress.style.display = 'none';
        ui.systemActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>Sinkronisasi data sedang berjalan...`;
        updateInterval = setInterval(() => {
            fetchData(`/update-status?pasaran=${currentPasaran}`).then(data => {
                if (!data) { clearInterval(updateInterval); return; }
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(updateInterval);
                    ui.updateDataBtn.disabled = false;
                    ui.updateDataBtn.innerHTML = ui.updateDataBtn.dataset.originalText;
                    if (data.status === 'completed') {
                       ui.systemActivityStatus.innerHTML = `<i class="fas fa-check-circle me-2"></i>${data.message}`;
                       showAlert('Sinkronisasi data selesai!', 'success');
                       updateSystemStatus();
                    } else {
                       ui.systemActivityStatus.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>Error: ${data.message}`;
                       showAlert('Sinkronisasi data gagal!', 'danger');
                    }
                }
            });
        }, 2000);
    }

    function startTraining() {
        const btn = ui.startTrainingBtn;
        btn.dataset.originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Melatih...`;
        ui.systemActivityStatus.textContent = 'Memulai proses training...';
        
        const formData = new FormData();
        formData.append('pasaran', currentPasaran);
        formData.append('training_mode', ui.trainingMode.value);
        // UPDATED: Mengirim status checkbox
        formData.append('use_recency_bias', ui.recencyBiasCheck.checked);
        
        fetchData('/start-training', { method: 'POST', body: formData }).then(data => {
            if (data?.status === 'success') {
                showAlert(data.message, 'info');
                monitorTrainingStatus();
            } else {
                btn.disabled = false;
                btn.innerHTML = btn.dataset.originalText;
                ui.systemActivityStatus.textContent = 'Gagal memulai training';
            }
        });
    }

    function monitorTrainingStatus() {
        if (trainingInterval) clearInterval(trainingInterval);
        ui.trainingProgress.style.display = 'flex';
        ui.trainingProgressBar.style.width = '0%';
        ui.trainingProgressBar.classList.remove('bg-success', 'bg-danger');
        ui.systemActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>Training sedang berjalan...`;
        let progress = 0;
        trainingInterval = setInterval(() => {
            fetchData(`/training-status?pasaran=${currentPasaran}`).then(data => {
                if (!data) { clearInterval(trainingInterval); return; }
                progress = Math.min(progress + 5, 95);
                ui.trainingProgressBar.style.width = `${progress}%`;
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(trainingInterval);
                    ui.trainingProgressBar.style.width = '100%';
                    ui.startTrainingBtn.disabled = false;
                    ui.startTrainingBtn.innerHTML = ui.startTrainingBtn.dataset.originalText;
                    if (data.status === 'completed') {
                        ui.trainingProgressBar.classList.add('bg-success');
                        ui.systemActivityStatus.innerHTML = `<i class="fas fa-check-circle me-2"></i>${data.message}`;
                        showAlert('Training selesai!', 'success');
                        updateSystemStatus();
                    } else {
                        ui.trainingProgressBar.classList.add('bg-danger');
                        ui.systemActivityStatus.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>Error: ${data.message}`;
                        showAlert('Training gagal!', 'danger');
                    }
                }
            });
        }, 2000);
    }
    
    function startEvaluation() {
        if (!ui.evalStartDate.value || !ui.evalEndDate.value) {
            showAlert('Silakan pilih tanggal mulai dan tanggal akhir.', 'warning');
            return;
        }
        if (new Date(ui.evalStartDate.value) > new Date(ui.evalEndDate.value)) {
            showAlert('Tanggal mulai tidak boleh setelah tanggal akhir.', 'warning');
            return;
        }
        const btn = ui.startEvaluationBtn;
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Mengevaluasi...`;
        updateActivePasaranDisplay();
        ui.evaluationResultArea.style.display = 'block';
        ui.evaluationStatus.innerHTML = `<div class="spinner-border text-info" role="status"></div><p class="mt-2">Menjalankan evaluasi...</p>`;
        ui.evaluationDetailTableBody.innerHTML = '';
        ui.evaluationSummaryCards.style.display = 'none';
        ui.retrainingRecommendation.style.display = 'none';
        const formData = new FormData();
        formData.append('pasaran', currentPasaran);
        formData.append('start_date', ui.evalStartDate.value);
        formData.append('end_date', ui.evalEndDate.value);
        fetchData('/start-evaluation', { method: 'POST', body: formData }).then(data => {
            if (data?.status === 'success') {
                showAlert(data.message, 'info');
                monitorEvaluationStatus();
            } else {
                btn.disabled = false;
                btn.innerHTML = `<i class="fas fa-play-circle me-2"></i>Mulai Evaluasi`;
                ui.evaluationStatus.textContent = 'Gagal memulai evaluasi.';
            }
        });
    }

    function validateEvaluationData(data) {
        if (!data || !data.data) return false;
        const { summary, results } = data.data;
        if (summary && summary.error) { console.warn('Evaluation summary contains error:', summary.error); return true; }
        if (results && Array.isArray(results)) {
            results.forEach((result, index) => {
                if (!result.kandidat_as || !result.kandidat_kop) { console.warn(`Missing data in result ${index}:`, result); }
            });
        }
        return true;
    }
    
    const highlightDigitsWithCorrectCommas = (predictions_str, actual_digit) => {
        if (!predictions_str) return '<span>-</span>';
        const digits = predictions_str.split(', ');
        return digits.map(digit => (digit === actual_digit) ? `<span class="digit-hit">${digit}</span>` : `<span>${digit}</span>`).join(' , '); 
    };

    const highlightAMWithCorrectCommas = (predictions_str, actual) => {
        if (!predictions_str) return '<span>-</span>';
        const predictedDigits = predictions_str.split(', ');
        return predictedDigits.map(digit => (actual && actual.includes(digit)) ? `<span class="digit-hit">${digit}</span>` : `<span>${digit}</span>`).join(' , '); 
    };

    const highlightCBWithCorrectCommas = (predictions_str, actual) => {
        if (!predictions_str) return '<span>-</span>';
        return (actual && actual.includes(predictions_str)) ? `<span class="digit-hit">${predictions_str}</span>` : `<span>${predictions_str}</span>`;
    };

    function monitorEvaluationStatus() {
        if (evaluationInterval) clearInterval(evaluationInterval);
        evaluationInterval = setInterval(() => {
            fetchData(`/evaluation-status?pasaran=${currentPasaran}`).then(data => {
                if (!data) { clearInterval(evaluationInterval); return; }
                if (data.status === 'completed') {
                    clearInterval(evaluationInterval);
                    ui.startEvaluationBtn.disabled = false;
                    ui.startEvaluationBtn.innerHTML = `<i class="fas fa-play-circle me-2"></i>Mulai Evaluasi`;
                    if (!validateEvaluationData(data)) {
                        ui.evaluationStatus.innerHTML = `<span class="badge bg-warning">Data Tidak Valid</span><p class="text-warning small mt-2">Data evaluasi tidak dalam format yang diharapkan</p>`;
                        return;
                    }
                    ui.evaluationStatus.innerHTML = `<span class="badge bg-success">Evaluasi Selesai</span>`;
                    const summary = data.data.summary;
                    const results = data.data.results;
                    if (summary && !summary.error) {
                        ui.evaluationSummaryCards.style.display = 'flex';
                        document.getElementById('evalTotalDays').textContent = summary.total_days_evaluated;
                        document.getElementById('evalAsAccuracy').textContent = `${(summary.as_accuracy * 100).toFixed(1)}%`;
                        document.getElementById('evalKopAccuracy').textContent = `${(summary.kop_accuracy * 100).toFixed(1)}%`;
                        document.getElementById('evalKepalaAccuracy').textContent = `${(summary.kepala_accuracy * 100).toFixed(1)}%`;
                        document.getElementById('evalEkorAccuracy').textContent = `${(summary.ekor_accuracy * 100).toFixed(1)}%`;
                        document.getElementById('evalAmAccuracy').textContent = `${(summary.am_accuracy * 100).toFixed(1)}%`;
                        document.getElementById('evalCbAccuracy').textContent = `${(summary.cb_accuracy * 100).toFixed(1)}%`;
                        if (summary.retraining_recommended) {
                            ui.retrainingRecommendation.textContent = `Rekomendasi: ${summary.retraining_reason}`;
                            ui.retrainingRecommendation.style.display = 'block';
                        } else {
                            ui.retrainingRecommendation.style.display = 'none';
                        }
                    } else if (summary && summary.error) {
                        ui.evaluationStatus.innerHTML += `<p class="text-danger mt-2">${summary.error}</p>`;
                    }
                    if (results && results.length > 0) {
                        let tableContent = '';
                        results.forEach(res => {
                            const actual = res.actual || '----';
                            tableContent += `<tr><td>${res.date}</td><td>${actual}</td><td>${highlightDigitsWithCorrectCommas(res.kandidat_as, actual[0])}</td><td>${highlightDigitsWithCorrectCommas(res.kandidat_kop, actual[1])}</td><td>${highlightDigitsWithCorrectCommas(res.kandidat_kepala, actual[2])}</td><td>${highlightDigitsWithCorrectCommas(res.kandidat_ekor, actual[3])}</td><td>${highlightAMWithCorrectCommas(res.angka_main, actual)}</td><td>${highlightCBWithCorrectCommas(res.colok_bebas, actual)}</td></tr>`;
                        });
                        ui.evaluationDetailTableBody.innerHTML = tableContent;
                    } else {
                        ui.evaluationDetailTableBody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">Tidak ada data evaluasi</td></tr>`;
                    }
                } else if (data.status === 'failed') {
                    clearInterval(evaluationInterval);
                    ui.startEvaluationBtn.disabled = false;
                    ui.startEvaluationBtn.innerHTML = `<i class="fas fa-play-circle me-2"></i>Mulai Evaluasi`;
                    ui.evaluationStatus.innerHTML = `<span class="badge bg-danger">Evaluasi Gagal</span><p class="text-danger small mt-2">${data.data?.error || 'Terjadi kesalahan'}</p>`;
                } else if (data.status === 'running') {
                    ui.evaluationStatus.innerHTML = `<div class="spinner-border text-info" role="status"></div><p class="mt-2">Evaluasi sedang berjalan...</p>`;
                }
            }).catch(error => {
                console.error('Error monitoring evaluation:', error);
                clearInterval(evaluationInterval);
                ui.startEvaluationBtn.disabled = false;
                ui.startEvaluationBtn.innerHTML = `<i class="fas fa-play-circle me-2"></i>Mulai Evaluasi`;
                ui.evaluationStatus.innerHTML = `<span class="badge bg-danger">Error</span><p class="text-danger small mt-2">Gagal memantau status evaluasi</p>`;
            });
        }, 2500);
    }

    function renderFeatureImportanceCharts(data) {
        const chartOptions = { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, title: { display: true, color: '#333', font: { size: 14 } } }, scales: { x: { beginAtZero: true, ticks: { color: '#666' } }, y: { ticks: { color: '#666', font: { size: 11 } } } } };
        ['as', 'kop', 'kepala', 'ekor'].forEach(digit => {
            const chartData = data[digit] || [];
            const ctx = document.getElementById(`${digit}Chart`).getContext('2d');
            if (charts[digit]) { charts[digit].destroy(); }
            charts[digit] = new Chart(ctx, { type: 'bar', data: { labels: chartData.map(d => d.feature), datasets: [{ label: 'Importance', data: chartData.map(d => d.weight), backgroundColor: 'rgba(13, 110, 253, 0.7)', borderColor: 'rgba(13, 110, 253, 1)', borderWidth: 1 }] }, options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { ...chartOptions.plugins.title, text: `Top Features - ${digit.toUpperCase()} (${chartData.length})` } } } });
        });
    }

    function loadFeatureImportance() {
        fetchData(`/feature-importance/${currentPasaran}`).then(data => {
            if (data) { renderFeatureImportanceCharts(data); showAlert('Feature importance diperbarui', 'success'); }
        });
    }

    function loadDriftLog() {
        fetchData('/drift-log').then(data => {
            if (data && Array.isArray(data)) { ui.driftLog.textContent = data.join(''); showAlert('Drift log diperbarui', 'success'); }
        });
    }

    function initializeHealthPanel() {
        ui.refreshFeatureImportance.addEventListener('click', loadFeatureImportance);
        ui.refreshDriftLog.addEventListener('click', loadDriftLog);
        loadFeatureImportance();
        loadDriftLog();
    }

    function initializeApp() {
        ui.pasaranSelect.addEventListener('change', () => {
            currentPasaran = ui.pasaranSelect.value;
            updateSystemStatus();
            updateActivePasaranDisplay();
            ui.predictionDisplay.style.display = 'none';
            ui.evaluationResultArea.style.display = 'none';
        });
        ui.predictBtn.addEventListener('click', getPrediction);
        ui.startTrainingBtn.addEventListener('click', startTraining);
        ui.updateDataBtn.addEventListener('click', startUpdateData);
        ui.startEvaluationBtn.addEventListener('click', startEvaluation);
        ui.evaluationTab.addEventListener('shown.bs.tab', () => { updateActivePasaranDisplay(); });
        ui.healthTab.addEventListener('shown.bs.tab', () => { updateActivePasaranDisplay(); initializeHealthPanel(); });
        const today = new Date();
        const tomorrow = new Date();
        tomorrow.setDate(today.getDate() + 1);
        const yesterday = new Date();
        yesterday.setDate(today.getDate() - 1);
        const aMonthAgo = new Date();
        aMonthAgo.setDate(today.getDate() - 30);
        ui.predictionDateInput.value = tomorrow.toISOString().split('T')[0];
        ui.evalEndDate.value = yesterday.toISOString().split('T')[0];
        ui.evalStartDate.value = aMonthAgo.toISOString().split('T')[0];
        updateSystemStatus();
        updateActivePasaranDisplay();
        ui.evaluationDetailTableBody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">Pilih tanggal dan klik "Mulai Evaluasi" untuk melihat data</td></tr>`;
    }
    
    initializeApp();
});
