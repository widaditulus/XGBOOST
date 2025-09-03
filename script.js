// script.js (Final - Siap Produksi dengan Analisis Kesalahan dan State UI yang Aman)

document.addEventListener('DOMContentLoaded', function() {
    // UPDATED: Plugin Chart.js didaftarkan satu kali di awal untuk stabilitas.
    Chart.register(ChartDataLabels);

    let currentPasaran = document.getElementById('pasaranSelect').value;
    let trainingInterval = null;
    let evaluationInterval = null;
    let updateInterval = null; 
    let tuningInterval = null;
    let cbTuningInterval = null;
    const charts = {};
    const cmCharts = {};

    const ui = {
        loadingOverlay: document.getElementById('loading-overlay'),
        pasaranSelect: document.getElementById('pasaranSelect'),
        trainingMode: document.getElementById('trainingMode'),
        startTrainingBtn: document.getElementById('startTrainingBtn'),
        updateDataBtn: document.getElementById('updateDataBtn'),
        recencyBiasCheck: document.getElementById('recencyBiasCheck'),
        dataInfo: document.getElementById('dataInfo'),
        dataRowCount: document.getElementById('dataRowCount'),
        dataStatus: document.getElementById('dataStatus'),
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
        startTuningBtn: document.getElementById('startTuningBtn'),
        startCbTuningBtn: document.getElementById('startCbTuningBtn'),
        tuningActivityStatus: document.getElementById('tuningActivityStatus'),
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
        activePasaranForHealth: document.getElementById('activePasaranForHealth'),
        showConfusionMatrixBtn: document.getElementById('showConfusionMatrixBtn'),
        confusionMatrixArea: document.getElementById('confusionMatrixArea')
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

    function checkDataStatus() {
        if (!currentPasaran) return;
        ui.dataStatus.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Memeriksa...`;
        fetchData(`/data-status/${currentPasaran}`).then(data => {
            if (!data || !data.status) {
                ui.dataStatus.innerHTML = `<i class="fas fa-times-circle text-danger"></i> Gagal memeriksa.`;
                return;
            };

            const btn = ui.updateDataBtn;
            btn.classList.remove('btn-info', 'btn-success', 'btn-warning');

            if (data.status === 'latest') {
                ui.dataStatus.innerHTML = `<i class="fas fa-check-circle text-success"></i> Data Terbaru (s/d ${data.local_date})`;
                btn.classList.add('btn-success');
            } else if (data.status === 'stale') {
                ui.dataStatus.innerHTML = `<i class="fas fa-exclamation-triangle text-warning"></i> Update Tersedia (Lokal s/d ${data.local_date})`;
                btn.classList.add('btn-warning');
            } else {
                ui.dataStatus.innerHTML = `<i class="fas fa-times-circle text-danger"></i> Error: ${data.message || 'Tidak diketahui'}`;
                btn.classList.add('btn-info');
            }
        });
    }

    function updateSystemStatus() {
        if (!currentPasaran) return;
        fetchData(`/debug/model-status/${currentPasaran}`).then(data => {
            if (!data) return;
            ui.modelStatus.innerHTML = data.models_ready ? 
                `<span class="badge bg-success">Siap</span>` : 
                `<span class="badge bg-danger">Perlu Training</span>`;
            ui.dataRowCount.innerHTML = data.data_manager_df_shape ? 
                `<i class="fas fa-database me-2"></i>${data.data_manager_df_shape[0]} baris data.` : 
                `<i class="fas fa-exclamation-triangle me-2"></i>Data tidak ditemukan.`;
        });
        checkDataStatus();
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
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span>`;
        ui.systemActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>Memulai sinkronisasi data...`;
        const formData = new FormData();
        formData.append('pasaran', currentPasaran);
        fetchData('/update-data', { method: 'POST', body: formData }).then(data => {
            if (data?.status === 'success') {
                showAlert(data.message, 'info');
                monitorUpdateStatus();
            } else {
                btn.disabled = false;
                btn.innerHTML = btn.dataset.originalText;
                ui.systemActivityStatus.textContent = 'Gagal memulai update.';
            }
        });
    }

    function monitorUpdateStatus() {
        if (updateInterval) clearInterval(updateInterval);
        ui.systemActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>Sinkronisasi data...`;
        updateInterval = setInterval(() => {
            fetchData(`/update-status?pasaran=${currentPasaran}`).then(data => {
                if (!data) { clearInterval(updateInterval); return; }
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(updateInterval);
                    ui.updateDataBtn.disabled = false;
                    ui.updateDataBtn.innerHTML = ui.updateDataBtn.dataset.originalText || '<i class="fas fa-sync-alt"></i> Update Data';
                    if (data.status === 'completed') {
                       ui.systemActivityStatus.innerHTML = `<i class="fas fa-check-circle me-2"></i>${data.message}`;
                       showAlert('Sinkronisasi data selesai!', 'success');
                       updateSystemStatus();
                    } else {
                       ui.systemActivityStatus.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>Error: ${data.message}`;
                    }
                }
            });
        }, 2000);
    }

    function startTraining() {
        const btn = ui.startTrainingBtn;
        btn.dataset.originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span>`;
        ui.systemActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>Memulai proses training...`;
        
        const formData = new FormData();
        formData.append('pasaran', currentPasaran);
        formData.append('training_mode', ui.trainingMode.value);
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
                    ui.startTrainingBtn.innerHTML = ui.startTrainingBtn.dataset.originalText || '<i class="fas fa-play"></i> Training';
                    if (data.status === 'completed') {
                        ui.trainingProgressBar.classList.add('bg-success');
                        ui.systemActivityStatus.innerHTML = `<i class="fas fa-check-circle me-2"></i>${data.message}`;
                        showAlert('Training selesai!', 'success');
                        updateSystemStatus();
                    } else {
                        ui.trainingProgressBar.classList.add('bg-danger');
                        ui.systemActivityStatus.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>Error: ${data.message}`;
                    }
                }
            });
        }, 2000);
    }
    
    function startTuning() {
        if (!confirm("Proses optimasi 4D akan memakan waktu lama dan sangat membebani CPU. Lanjutkan?")) {
            return;
        }
        const btn = ui.startTuningBtn;
        btn.disabled = true;
        ui.tuningActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>Memulai optimasi 4D...`;
        
        const formData = new FormData();
        formData.append('pasaran', currentPasaran);
        
        fetchData('/start-tuning', { method: 'POST', body: formData }).then(data => {
            if (data?.status === 'success') {
                showAlert(data.message, 'info');
                monitorTuningStatus();
            } else {
                btn.disabled = false;
                ui.tuningActivityStatus.textContent = 'Gagal memulai optimasi 4D';
                showAlert(data?.message || 'Gagal memulai optimasi 4D', 'danger');
            }
        });
    }

    function monitorTuningStatus() {
        if (tuningInterval) clearInterval(tuningInterval);
        ui.tuningActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>Optimasi 4D berjalan...`;
        tuningInterval = setInterval(() => {
            fetchData(`/tuning-status?pasaran=${currentPasaran}`).then(data => {
                if (!data) { clearInterval(tuningInterval); return; }
                
                ui.tuningActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>${data.message}`;

                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(tuningInterval);
                    ui.startTuningBtn.disabled = false;
                    ui.tuningActivityStatus.innerHTML = (data.status === 'completed') ? 
                        `<i class="fas fa-check-circle me-2 text-success"></i>${data.message}` : 
                        `<i class="fas fa-exclamation-triangle me-2 text-danger"></i>Error: ${data.message}`;
                    showAlert(data.message, data.status === 'completed' ? 'success' : 'danger');
                }
            });
        }, 5000);
    }

    function startCbTuning() {
        if (!confirm("Proses optimasi CB akan memakan waktu lama (bisa lebih dari 1 jam) dan sangat membebani CPU. Lanjutkan?")) {
            return;
        }
        const btn = ui.startCbTuningBtn;
        btn.disabled = true;
        ui.tuningActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>Memulai optimasi CB...`;
        
        const formData = new FormData();
        formData.append('pasaran', currentPasaran);
        
        fetchData('/start-tuning-cb', { method: 'POST', body: formData }).then(data => {
            if (data?.status === 'success') {
                showAlert(data.message, 'info');
                monitorCbTuningStatus();
            } else {
                btn.disabled = false;
                ui.tuningActivityStatus.textContent = 'Gagal memulai optimasi CB';
                showAlert(data?.message || 'Gagal memulai optimasi CB', 'danger');
            }
        });
    }

    function monitorCbTuningStatus() {
        if (cbTuningInterval) clearInterval(cbTuningInterval);
        ui.tuningActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>Optimasi CB berjalan...`;
        cbTuningInterval = setInterval(() => {
            fetchData(`/cb-tuning-status?pasaran=${currentPasaran}`).then(data => {
                if (!data) { clearInterval(cbTuningInterval); return; }
                
                ui.tuningActivityStatus.innerHTML = `<i class="fas fa-sync-alt fa-spin me-2"></i>${data.message}`;

                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(cbTuningInterval);
                    ui.startCbTuningBtn.disabled = false;
                    ui.tuningActivityStatus.innerHTML = (data.status === 'completed') ? 
                        `<i class="fas fa-check-circle me-2 text-success"></i>${data.message}` : 
                        `<i class="fas fa-exclamation-triangle me-2 text-danger"></i>Error: ${data.message}`;
                    showAlert(data.message, data.status === 'completed' ? 'success' : 'danger');
                }
            });
        }, 5000);
    }

    function startEvaluation() {
        if (!ui.evalStartDate.value || !ui.evalEndDate.value) {
            showAlert('Silakan pilih tanggal mulai dan tanggal akhir.', 'warning');
            return;
        }
        const btn = ui.startEvaluationBtn;
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span>`;
        updateActivePasaranDisplay();
        ui.evaluationResultArea.style.display = 'block';
        ui.evaluationStatus.innerHTML = `<div class="spinner-border text-info" role="status"></div><p class="mt-2">Menjalankan evaluasi...</p>`;
        ui.evaluationDetailTableBody.innerHTML = '';
        ui.evaluationSummaryCards.style.display = 'none';
        ui.showConfusionMatrixBtn.style.display = 'none';
        ui.confusionMatrixArea.style.display = 'none';
        
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
                    ui.evaluationStatus.innerHTML = `<span class="badge bg-success">Selesai</span>`;
                    
                    if (data.data.results && data.data.results.length > 0) {
                        ui.showConfusionMatrixBtn.style.display = 'inline-block';
                    }

                    const summary = data.data.summary;
                    const results = data.data.results;
                    
                    if (summary && !summary.error) {
                        ui.evaluationSummaryCards.style.display = 'flex';
                        ui.evalTotalDays.textContent = summary.total_days_evaluated;
                        ['as', 'kop', 'kepala', 'ekor', 'am', 'cb'].forEach(key => {
                            const elem = document.getElementById(`eval${key.charAt(0).toUpperCase() + key.slice(1)}Accuracy`);
                            if(elem) elem.textContent = `${(summary[`${key}_accuracy`] * 100).toFixed(1)}%`;
                        });
                    }
                    
                    if (results && results.length > 0) {
                        let tableContent = '';
                        results.forEach(res => {
                            const actual = res.actual || '----';
                            tableContent += `<tr>
                                <td>${res.date}</td>
                                <td>${actual}</td>
                                <td>${highlightDigitsWithCorrectCommas(res.kandidat_as, actual[0])}</td>
                                <td>${highlightDigitsWithCorrectCommas(res.kandidat_kop, actual[1])}</td>
                                <td>${highlightDigitsWithCorrectCommas(res.kandidat_kepala, actual[2])}</td>
                                <td>${highlightDigitsWithCorrectCommas(res.kandidat_ekor, actual[3])}</td>
                                <td>${highlightAMWithCorrectCommas(res.angka_main, actual)}</td>
                                <td>${highlightCBWithCorrectCommas(res.colok_bebas, actual)}</td>
                            </tr>`;
                        });
                        ui.evaluationDetailTableBody.innerHTML = tableContent;
                    } else {
                        ui.evaluationDetailTableBody.innerHTML = `<tr><td colspan="8">Tidak ada hasil untuk ditampilkan.</td></tr>`;
                    }
                } else if (data.status === 'failed') {
                    clearInterval(evaluationInterval);
                    ui.startEvaluationBtn.disabled = false;
                    ui.startEvaluationBtn.innerHTML = `<i class="fas fa-play-circle me-2"></i>Mulai Evaluasi`;
                    ui.evaluationStatus.innerHTML = `<span class="badge bg-danger">Gagal</span>: ${data.data?.summary?.error || 'Error tidak diketahui'}`;
                }
            });
        }, 2500);
    }

    function renderConfusionMatrix(data) {
        if (!data || !data.labels || !data.matrices) {
            showAlert('Gagal memuat data confusion matrix.', 'danger');
            return;
        }

        const { labels, matrices } = data;
        const createDataset = (matrix) => {
            const dataset = [];
            for (let i = 0; i < matrix.length; i++) {
                for (let j = 0; j < matrix[i].length; j++) {
                    dataset.push({ x: labels[j], y: labels[i], v: matrix[i][j] });
                }
            }
            return dataset;
        };
        
        Object.values(cmCharts).forEach(chart => {
            if (chart) chart.destroy();
        });

        ['as', 'kop', 'kepala', 'ekor'].forEach(digit => {
            const chartId = `cm${digit.charAt(0).toUpperCase() + digit.slice(1)}Chart`;
            const ctx = document.getElementById(chartId).getContext('2d');
            cmCharts[digit] = new Chart(ctx, {
                type: 'matrix',
                data: {
                    datasets: [{
                        label: `CM ${digit}`, data: createDataset(matrices[digit]),
                        backgroundColor: (ctx) => {
                            const value = ctx.dataset.data[ctx.dataIndex].v;
                            if (ctx.raw.x === ctx.raw.y) return 'rgba(25, 135, 84, 0.7)';
                            if (value === 0) return 'rgba(248, 249, 250, 1)';
                            const alpha = Math.min(0.2 + (value / 5), 1);
                            return `rgba(220, 53, 69, ${alpha})`;
                        },
                        borderColor: 'grey', borderWidth: 1,
                        width: ({chart}) => (chart.chartArea.width / labels.length) - 1,
                        height: ({chart}) =>(chart.chartArea.height / labels.length) - 1
                    }]
                },
                options: {
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                title: () => '',
                                label: (ctx) => `Aktual: ${ctx.raw.y}, Prediksi: ${ctx.raw.x}, Jumlah: ${ctx.raw.v}`
                            }
                        },
                        datalabels: {
                            formatter: (context) => context.dataset.data[context.dataIndex].v > 0 ? context.dataset.data[context.dataIndex].v : '',
                            color: (context) => {
                                 const value = context.dataset.data[context.dataIndex].v;
                                 if (context.raw.x === context.raw.y || value > 5) return 'white';
                                 return 'black';
                            },
                            font: { size: '10px' }
                        }
                    },
                    scales: {
                        x: { type: 'category', labels: labels, grid: { display: false } },
                        // UPDATED: 'offset: true' dihapus untuk memperbaiki render matriks.
                        y: { type: 'category', labels: labels, grid: { display: false } }
                    },
                    responsive: true, maintainAspectRatio: false
                }
            });
        });
        ui.confusionMatrixArea.style.display = 'block';
    }

    function getAndShowConfusionMatrix() {
        const btn = ui.showConfusionMatrixBtn;
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Memuat...`;
        
        fetchData(`/confusion-matrix/${currentPasaran}`).then(data => {
            if (data && !data.error) {
                renderConfusionMatrix(data);
            } else {
                 showAlert(data.error || 'Gagal mengambil data', 'danger');
            }
        }).finally(() => {
            btn.disabled = false;
            btn.innerHTML = `<i class="fas fa-th me-2"></i>Tampilkan Analisis Kesalahan`;
        });
    }

    function renderFeatureImportanceCharts(data) {
        const chartOptions = { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } };
        ['as', 'kop', 'kepala', 'ekor'].forEach(digit => {
            const chartData = data[digit] || [];
            const ctx = document.getElementById(`${digit}Chart`)?.getContext('2d');
            if(!ctx) return;
            if (charts[digit]) { charts[digit].destroy(); }
            charts[digit] = new Chart(ctx, { type: 'bar', data: { labels: chartData.map(d => d.feature), datasets: [{ data: chartData.map(d => d.weight), backgroundColor: 'rgba(13, 110, 253, 0.7)' }] }, options: chartOptions });
        });
    }

    function loadFeatureImportance() {
        fetchData(`/feature-importance/${currentPasaran}`).then(data => {
            if (data) renderFeatureImportanceCharts(data);
        });
    }

    function loadDriftLog() {
        fetchData('/drift-log').then(data => {
            if (data && Array.isArray(data)) { ui.driftLog.textContent = data.join(''); }
        });
    }

    async function initializeApp() {
        ui.pasaranSelect.addEventListener('change', () => {
            currentPasaran = ui.pasaranSelect.value;
            updateSystemStatus();
            updateActivePasaranDisplay();
            ui.predictionDisplay.style.display = 'none';
            ui.evaluationResultArea.style.display = 'none'; 
            ui.showConfusionMatrixBtn.style.display = 'none';
            ui.confusionMatrixArea.style.display = 'none';
        });

        ui.predictBtn.addEventListener('click', getPrediction);
        ui.startTrainingBtn.addEventListener('click', startTraining);
        ui.updateDataBtn.addEventListener('click', startUpdateData);
        ui.startTuningBtn.addEventListener('click', startTuning);
        ui.startCbTuningBtn.addEventListener('click', startCbTuning);
        ui.startEvaluationBtn.addEventListener('click', startEvaluation);
        ui.showConfusionMatrixBtn.addEventListener('click', getAndShowConfusionMatrix);
        
        ui.evaluationTab.addEventListener('shown.bs.tab', updateActivePasaranDisplay);
        ui.healthTab.addEventListener('shown.bs.tab', () => {
            updateActivePasaranDisplay();
            loadFeatureImportance();
            loadDriftLog();
        });

        ui.refreshFeatureImportance.addEventListener('click', loadFeatureImportance);
        ui.refreshDriftLog.addEventListener('click', loadDriftLog);
        
        const today = new Date();
        const tomorrow = new Date(); tomorrow.setDate(today.getDate() + 1);
        const yesterday = new Date(); yesterday.setDate(today.getDate() - 1);
        const aMonthAgo = new Date(); aMonthAgo.setDate(today.getDate() - 30);
        ui.predictionDateInput.value = tomorrow.toISOString().split('T')[0];
        ui.evalEndDate.value = yesterday.toISOString().split('T')[0];
        ui.evalStartDate.value = aMonthAgo.toISOString().split('T')[0];
        
        await updateSystemStatus();
        updateActivePasaranDisplay();
        ui.loadingOverlay.classList.add('hidden');
    }
    
    initializeApp();
});