import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import signal
import pickle
from threading import Lock, Event, Semaphore
import threading
import sys

try:
    import PyPDF2
except ImportError:
    st.error("PyPDF2 non installé. Exécutez: pip install PyPDF2")
    st.stop()

try:
    import openai
except ImportError:
    st.error("OpenAI non installé. Exécutez: pip install openai")
    st.stop()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("python-dotenv non installé. Les variables .env ne seront pas chargées.")

st.set_page_config(
    page_title="PDF Analyzer - Rate Limited",
    page_icon="📄",
    layout="wide"
)

# Configuration du rate limiting
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 15
    max_concurrent_requests: int = 15
    retry_attempts: int = 3
    initial_backoff: float = 1.0
    backoff_multiplier: float = 2.0
    timeout_seconds: int = 60
    error_threshold_pct: float = 10.0
    request_interval_seconds: float = 4.0  # 15 req/min = 1 req/4s
    openai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.3

# Métriques de traitement
@dataclass
class ProcessingMetrics:
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    rate_limit_errors: int = 0
    total_processing_time: float = 0
    total_tokens: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    chunk_times: List[float] = field(default_factory=list)
    error_details: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return (self.completed_chunks / self.total_chunks) * 100
    
    @property
    def error_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return (self.rate_limit_errors / self.total_chunks) * 100
    
    @property
    def avg_chunk_time(self) -> float:
        if not self.chunk_times:
            return 0.0
        return sum(self.chunk_times) / len(self.chunk_times)

# Gestionnaire de taux avec cadence contrôlée
class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.semaphore = Semaphore(config.max_concurrent_requests)
        self.last_request_time = 0
        self.lock = Lock()
        self.request_times = []
        self.shutdown_event = Event()
        
    def wait_for_slot(self):
        """Attend le prochain créneau disponible selon la cadence"""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.config.request_interval_seconds:
                wait_time = self.config.request_interval_seconds - time_since_last
                if not self.shutdown_event.is_set():
                    time.sleep(wait_time)
            
            self.last_request_time = time.time()
            self.request_times.append(self.last_request_time)
            
            # Nettoyer les anciennes entrées
            cutoff = now - 60
            self.request_times = [t for t in self.request_times if t > cutoff]
    
    def acquire(self):
        """Acquiert un slot pour une requête"""
        return self.semaphore.acquire(blocking=True, timeout=1)
    
    def release(self):
        """Libère un slot après requête"""
        self.semaphore.release()
    
    def shutdown(self):
        """Arrêt gracieux"""
        self.shutdown_event.set()

# Gestionnaire de chunks avec persistance
class ChunkManager:
    def __init__(self, state_file: str = "chunk_state.pkl"):
        self.state_file = state_file
        self.processed_chunks: Set[str] = set()
        self.failed_chunks: Dict[str, int] = {}
        self.lock = Lock()
        self.load_state()
    
    def load_state(self):
        """Charge l'état depuis le fichier"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.processed_chunks = state.get('processed', set())
                    self.failed_chunks = state.get('failed', {})
        except Exception as e:
            logging.error(f"Erreur chargement état: {e}")
    
    def save_state(self):
        """Sauvegarde l'état dans le fichier"""
        with self.lock:
            try:
                state = {
                    'processed': self.processed_chunks,
                    'failed': self.failed_chunks
                }
                with open(self.state_file, 'wb') as f:
                    pickle.dump(state, f)
            except Exception as e:
                logging.error(f"Erreur sauvegarde état: {e}")
    
    def mark_processed(self, chunk_id: str):
        """Marque un chunk comme traité"""
        with self.lock:
            self.processed_chunks.add(chunk_id)
            if chunk_id in self.failed_chunks:
                del self.failed_chunks[chunk_id]
            self.save_state()
    
    def mark_failed(self, chunk_id: str):
        """Marque un chunk comme échoué"""
        with self.lock:
            self.failed_chunks[chunk_id] = self.failed_chunks.get(chunk_id, 0) + 1
            self.save_state()
    
    def is_processed(self, chunk_id: str) -> bool:
        """Vérifie si un chunk a déjà été traité"""
        with self.lock:
            return chunk_id in self.processed_chunks
    
    def get_retry_count(self, chunk_id: str) -> int:
        """Obtient le nombre de tentatives pour un chunk"""
        with self.lock:
            return self.failed_chunks.get(chunk_id, 0)

# Processeur de chunks avec retry intelligent
class ChunkProcessor:
    def __init__(self, api_key: str, config: RateLimitConfig, metrics: ProcessingMetrics):
        self.api_key = api_key
        self.config = config
        self.metrics = metrics
        self.rate_limiter = RateLimiter(config)
        self.chunk_manager = ChunkManager()
        self.shutdown_flag = False
        openai.api_key = api_key
        
    def process_chunk_with_retry(self, chunk_data: Dict) -> Optional[Dict]:
        """Traite un chunk avec retry et backoff"""
        chunk_id = chunk_data['id']
        
        # Vérifier si déjà traité
        if self.chunk_manager.is_processed(chunk_id):
            add_log(f"Chunk {chunk_id} déjà traité, skip", "info")
            return None
        
        retry_count = self.chunk_manager.get_retry_count(chunk_id)

        for attempt in range(retry_count, self.config.retry_attempts):
            if self.shutdown_flag:
                add_log(f"Arrêt demandé, abandon chunk {chunk_id}", "warning")
                return None
            
            try:
                # Attendre le créneau
                self.rate_limiter.wait_for_slot()
                
                # Acquérir un slot concurrent
                if not self.rate_limiter.acquire():
                    add_log(f"Timeout acquisition slot pour chunk {chunk_id}", "warning")
                    continue
                
                start_time = time.time()
                
                try:
                    # Appel API OpenAI
                    response = openai.ChatCompletion.create(
                        model=self.config.openai_model,
                        messages=[
                            {"role": "system", "content": "Vous êtes un assistant d'analyse de documents."},
                            {"role": "user", "content": chunk_data['content']}
                        ],
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        timeout=self.config.timeout_seconds
                    )
                    
                    # Traitement réussi
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    self.metrics.completed_chunks += 1
                    self.metrics.chunk_times.append(processing_time)
                    self.metrics.total_tokens += response['usage']['total_tokens']
                    
                    self.chunk_manager.mark_processed(chunk_id)
                    
                    add_log(f"Chunk {chunk_id} traité en {processing_time:.2f}s", "success")
                    
                    return {
                        'chunk_id': chunk_id,
                        'response': response['choices'][0]['message']['content'],
                        'tokens': response['usage']['total_tokens'],
                        'processing_time': processing_time
                    }
                    
                finally:
                    self.rate_limiter.release()
                    
            except openai.error.RateLimitError as e:
                # Erreur 429
                self.metrics.rate_limit_errors += 1
                self.metrics.error_details['rate_limit'] = self.metrics.error_details.get('rate_limit', 0) + 1

                self.chunk_manager.mark_failed(chunk_id)
                
                backoff_time = self.config.initial_backoff * (self.config.backoff_multiplier ** attempt)
                add_log(f"Rate limit chunk {chunk_id}, backoff {backoff_time}s", "warning")
                
                if not self.shutdown_flag:
                    time.sleep(backoff_time)
                    
            except openai.error.Timeout as e:
                self.metrics.error_details['timeout'] = self.metrics.error_details.get('timeout', 0) + 1
                add_log(f"Timeout chunk {chunk_id}, tentative {attempt + 1}", "error")
                self.chunk_manager.mark_failed(chunk_id)
                
            except Exception as e:
                self.metrics.error_details['other'] = self.metrics.error_details.get('other', 0) + 1
                add_log(f"Erreur chunk {chunk_id}: {str(e)}", "error")
                self.chunk_manager.mark_failed(chunk_id)
        
        # Échec après toutes les tentatives
        self.metrics.failed_chunks += 1
        add_log(f"Chunk {chunk_id} échoué après {self.config.retry_attempts} tentatives", "error")
        
        return None
    
    def process_batch(self, chunks: List[Dict], progress_callback=None):
        """Traite un batch de chunks avec pool de threads"""
        self.metrics.total_chunks = len(chunks)
        self.metrics.start_time = datetime.now()
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
            # Soumettre les tâches
            future_to_chunk = {
                executor.submit(self.process_chunk_with_retry, chunk): chunk 
                for chunk in chunks
            }
            
            # Traiter les résultats
            for future in as_completed(future_to_chunk):
                if self.shutdown_flag:
                    executor.shutdown(wait=False)
                    break
                
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                    if progress_callback:
                        progress = (self.metrics.completed_chunks + self.metrics.failed_chunks) / self.metrics.total_chunks
                        progress_callback(progress, chunk['id'])
                        
                except Exception as e:
                    add_log(f"Erreur traitement chunk {chunk['id']}: {str(e)}", "error")
                
                # Vérifier le seuil d'erreur
                if self.metrics.error_rate > self.config.error_threshold_pct:
                    add_log(f"Seuil d'erreur dépassé ({self.metrics.error_rate:.1f}%), arrêt", "critical")
                    self.shutdown()
                    break
        
        self.metrics.end_time = datetime.now()
        self.metrics.total_processing_time = (self.metrics.end_time - self.metrics.start_time).total_seconds()
        
        return results
    
    def shutdown(self):
        """Arrêt gracieux du processeur"""
        add_log("Arrêt gracieux du processeur", "info")
        self.shutdown_flag = True
        self.rate_limiter.shutdown()
        self.chunk_manager.save_state()

# Extracteur PDF amélioré
def extract_pdf_chunks(file, chunk_size: int = 1000) -> List[Dict]:
    """Extrait le texte du PDF et le divise en chunks"""
    chunks = []
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = ""
        
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
        
        # Diviser en chunks
        words = full_text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_id = hashlib.md5(f"{file.name}_{i}".encode()).hexdigest()[:8]
            
            chunks.append({
                'id': chunk_id,
                'file': file.name,
                'index': i // chunk_size,
                'content': chunk_text
            })
            
    except Exception as e:
        add_log(f"Erreur extraction PDF {file.name}: {str(e)}", "error")
    
    return chunks

# État de session
def init_session_state():
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = ProcessingMetrics()

def add_log(message: str, level: str = "info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        'timestamp': timestamp,
        'level': level,
        'message': message
    })

def load_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    return None

# Gestionnaire de signaux pour arrêt gracieux
def signal_handler(signum, frame):
    if st.session_state.processor:
        st.session_state.processor.shutdown()
    sys.exit(0)

# Register signal handlers only in the main thread to avoid ValueError in
# environments like Streamlit that execute code in a separate thread.
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
else:
    logging.warning("Signal handlers not registered because not running in the main thread")

def main():
    init_session_state()
    
    st.title("📄 PDF Analyzer avec Rate Limiting")
    st.markdown("### Version avec gestion intelligente des quotas OpenAI")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Clé API
        api_key_from_env = load_api_key()
        
        if api_key_from_env:
            st.success("✅ Clé API chargée depuis .env")
            api_key = api_key_from_env
            st.text("Clé API: " + api_key[:10] + "..." + api_key[-4:])
        else:
            st.warning("⚠️ Clé API non trouvée dans .env")
            api_key = st.text_input("Clé API manuelle", type="password", 
                                   help="Votre clé API OpenAI")
        
        # Configuration rate limiting
        st.markdown("#### 🎛️ Rate Limiting")
        
        col1, col2 = st.columns(2)
        with col1:
            requests_per_minute = st.number_input(
                "Requêtes/minute", min_value=5, max_value=60, value=15
            )
        with col2:
            max_concurrent = st.number_input(
                "Pool concurrent", min_value=5, max_value=50, value=15
            )
        
        col3, col4 = st.columns(2)
        with col3:
            retry_attempts = st.number_input(
                "Tentatives retry", min_value=1, max_value=10, value=3
            )
        with col4:
            error_threshold = st.slider(
                "Seuil erreur %", min_value=5, max_value=50, value=10
            )
        
        # Taille des chunks
        chunk_size = st.number_input(
            "Taille chunks (mots)", min_value=100, max_value=5000, value=1000
        )

        st.markdown("#### 🤖 OpenAI")
        openai_model = st.text_input("Modèle", value="gpt-3.5-turbo")
        max_tokens_setting = st.number_input(
            "Max tokens", min_value=50, max_value=4000, value=500, step=50
        )
        temperature_setting = st.slider(
            "Température", min_value=0.0, max_value=1.0, value=0.3, step=0.1
        )
    
    # Interface principale
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Fichiers", "🚀 Analyse", "📊 Monitoring", "📋 Logs"])
    
    with tab1:
        st.header("Upload de fichiers PDF")
        
        uploaded_files = st.file_uploader(
            "Sélectionnez vos fichiers PDF",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} fichier(s) PDF chargé(s)")
            
            total_chunks = 0
            for file in uploaded_files:
                size_mb = file.size / (1024 * 1024)
                # Estimation du nombre de chunks
                estimated_chunks = max(1, int(size_mb * 10))  # Approximation
                total_chunks += estimated_chunks
                st.write(f"📄 {file.name} ({size_mb:.1f} MB) - ~{estimated_chunks} chunks")
            
            st.info(f"📊 Estimation totale: ~{total_chunks} chunks à traiter")
    
    with tab2:
        st.header("Analyse avec Rate Limiting")
        
        if not api_key:
            st.error("❌ Clé API OpenAI requise")
        elif not uploaded_files:
            st.info("📁 Veuillez d'abord uploader des fichiers PDF")
        else:
            st.success("✅ Prêt pour l'analyse")
            
            # Configuration affichée
            st.markdown("### 🎛️ Paramètres Rate Limiting")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Requêtes/minute", requests_per_minute)
                st.caption(f"Intervalle: {60/requests_per_minute:.1f}s")
            with col2:
                st.metric("Pool concurrent", max_concurrent)
            with col3:
                st.metric("Tentatives retry", retry_attempts)
            with col4:
                st.metric("Seuil erreur", f"{error_threshold}%")

            st.markdown("### 🤖 Paramètres OpenAI")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Modèle", openai_model)
            with mcol2:
                st.metric("Max tokens", max_tokens_setting)
            with mcol3:
                st.metric("Température", temperature_setting)
            
            # Boutons de contrôle
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Démarrer l'analyse", type="primary"):
                    # Configuration
                    config = RateLimitConfig(
                        requests_per_minute=requests_per_minute,
                        max_concurrent_requests=max_concurrent,
                        retry_attempts=retry_attempts,
                        error_threshold_pct=error_threshold,
                        request_interval_seconds=60/requests_per_minute,
                        openai_model=openai_model,
                        max_tokens=max_tokens_setting,
                        temperature=temperature_setting
                    )
                    
                    # Réinitialiser les métriques
                    st.session_state.metrics = ProcessingMetrics()
                    
                    # Créer le processeur
                    st.session_state.processor = ChunkProcessor(
                        api_key=api_key,
                        config=config,
                        metrics=st.session_state.metrics
                    )
                    
                    # Extraire tous les chunks
                    all_chunks = []
                    for file in uploaded_files:
                        chunks = extract_pdf_chunks(file, chunk_size)
                        all_chunks.extend(chunks)
                    
                    add_log(f"Démarrage du traitement de {len(all_chunks)} chunks", "info")
                    
                    # Conteneurs pour l'affichage
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    metrics_container = st.container()
                    
                    def update_progress(progress, chunk_id):
                        progress_bar.progress(progress)
                        status_text.text(f"Traitement chunk {chunk_id}... ({progress*100:.1f}%)")
                        
                        # Mise à jour des métriques en temps réel
                        with metrics_container:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Complétés", st.session_state.metrics.completed_chunks)
                            with col2:
                                st.metric("Échoués", st.session_state.metrics.failed_chunks)
                            with col3:
                                st.metric("Erreurs 429", st.session_state.metrics.rate_limit_errors)
                            with col4:
                                if st.session_state.metrics.chunk_times:
                                    st.metric("Temps moy.", f"{st.session_state.metrics.avg_chunk_time:.1f}s")
                    
                    # Lancer le traitement
                    try:
                        results = st.session_state.processor.process_batch(
                            all_chunks, 
                            progress_callback=update_progress
                        )
                        
                        st.session_state.results = results
                        
                        status_text.text("✅ Analyse terminée !")
                        st.success(f"🎉 Traitement terminé: {len(results)} chunks réussis")
                        
                        # Sauvegarder les résultats
                        if st.button("💾 Exporter les résultats"):
                            export_results(results)
                            
                    except Exception as e:
                        st.error(f"❌ Erreur pendant le traitement: {str(e)}")
                        add_log(f"Erreur fatale: {str(e)}", "critical")
            
            with col2:
                if st.session_state.processor and st.button("⏸️ Pause", type="secondary"):
                    st.session_state.processor.shutdown()
                    st.warning("⏸️ Traitement en pause")
            
            with col3:
                if st.button("🔄 Reprendre", type="secondary"):
                    st.info("La reprise utilisera l'état sauvegardé")
    
    with tab3:
        st.header("📈 Monitoring Rate Limiting")
        
        if st.session_state.metrics.total_chunks > 0:
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Progression", 
                    f"{st.session_state.metrics.completed_chunks}/{st.session_state.metrics.total_chunks}",
                    f"{st.session_state.metrics.success_rate:.1f}%"
                )
            with col2:
                st.metric(
                    "Taux de succès", 
                    f"{st.session_state.metrics.success_rate:.1f}%",
                    delta=f"-{st.session_state.metrics.error_rate:.1f}% erreurs"
                )
            with col3:
                st.metric(
                    "Erreurs 429", 
                    st.session_state.metrics.rate_limit_errors,
                    f"{st.session_state.metrics.error_rate:.1f}%"
                )
            with col4:
                if st.session_state.metrics.chunk_times:
                    st.metric(
                        "Temps moyen", 
                        f"{st.session_state.metrics.avg_chunk_time:.2f}s",
                        f"Total: {st.session_state.metrics.total_processing_time:.0f}s"
                    )
            
            # Détails des erreurs
            if st.session_state.metrics.error_details:
                st.markdown("#### 🚨 Détail des erreurs")
                error_df = pd.DataFrame(
                    list(st.session_state.metrics.error_details.items()),
                    columns=['Type', 'Count']
                )
                st.dataframe(error_df)
            
            # Graphique de performance
            if st.session_state.metrics.chunk_times:
                st.markdown("#### 📊 Performance par chunk")
                chart_data = pd.DataFrame({
                    'Chunk': range(len(st.session_state.metrics.chunk_times)),
                    'Temps (s)': st.session_state.metrics.chunk_times
                })
                st.line_chart(chart_data.set_index('Chunk'))
            
            # Statistiques tokens
            if st.session_state.metrics.total_tokens > 0:
                st.markdown("#### 🪙 Consommation de tokens")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total tokens", f"{st.session_state.metrics.total_tokens:,}")
                with col2:
                    avg_tokens = st.session_state.metrics.total_tokens / max(1, st.session_state.metrics.completed_chunks)
                    st.metric("Moy. par chunk", f"{avg_tokens:.0f}")
                with col3:
                    # Estimation coût (GPT-3.5-turbo: $0.002 per 1K tokens)
                    cost = (st.session_state.metrics.total_tokens / 1000) * 0.002
                    st.metric("Coût estimé", f"${cost:.4f}")
        else:
            st.info("📊 Les métriques apparaîtront pendant le traitement")
    
    with tab4:
        st.header("📋 Journal d'activité")
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        with col1:
            log_level = st.selectbox(
                "Niveau", 
                ["Tous", "info", "success", "warning", "error", "critical"],
                index=0
            )
        with col2:
            if st.button("🗑️ Vider les logs"):
                st.session_state.logs = []
                st.success("Logs vidés")
        with col3:
            auto_refresh = st.checkbox("Auto-refresh", value=True)
        
        # Affichage des logs
        if st.session_state.logs:
            # Filtrer si nécessaire
            filtered_logs = st.session_state.logs
            if log_level != "Tous":
                filtered_logs = [log for log in filtered_logs if log['level'] == log_level]
            
            # Afficher en ordre inverse (plus récent en haut)
            for log in reversed(filtered_logs[-100:]):  # Limiter à 100 derniers
                level_emoji = {
                    'info': 'ℹ️',
                    'success': '✅',
                    'warning': '⚠️',
                    'error': '❌',
                    'critical': '🚨'
                }.get(log['level'], '📝')
                
                st.text(f"[{log['timestamp']}] {level_emoji} {log['message']}")
            
            # Auto-refresh
            if auto_refresh:
                time.sleep(1)
                st.rerun()
        else:
            st.info("Aucun log pour le moment")

def export_results(results: List[Dict]):
    """Exporte les résultats dans un fichier"""
    try:
        # Créer le dossier Exports
        export_dir = Path("Exports")
        export_dir.mkdir(exist_ok=True)
        
        # Nom du fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = export_dir / f"results_{timestamp}.json"
        
        # Sauvegarder les résultats
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        st.success(f"✅ Résultats exportés: {filename}")
        add_log(f"Export réussi: {filename}", "success")
        
    except Exception as e:
        st.error(f"❌ Erreur export: {str(e)}")
        add_log(f"Erreur export: {str(e)}", "error")

if __name__ == "__main__":
    main()
