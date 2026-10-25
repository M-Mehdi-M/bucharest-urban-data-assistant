'use strict';
document.addEventListener('DOMContentLoaded', () => {
  const byId = id => document.getElementById(id);
  const input = byId('user-input');
  const send = byId('send-button');
  const chat = byId('chat-container');
  let ready = document.body.dataset.ready === 'true';
  let busy = false;
  let polling = false;
  let snapshot = null;
  let resetSession = false;
  const notice = text => {
    byId('service-notice').textContent = text || '';
    byId('service-notice').hidden = !text;
  };
  function message(text, role = 'assistant') {
    const item = document.createElement('div');
    item.className = `message ${role}`;
    const label = document.createElement('span');
    label.className = 'message-label';
    label.textContent = role === 'user' ? 'Tu' : 'Asistent';
    item.append(label, document.createTextNode(String(text)));
    chat.append(item);
    chat.scrollTop = chat.scrollHeight;
    return item;
  }
  function showEvidence(item, data) {
    const evidence = data.evidence || {};
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = 'Surse și limite';
    details.append(summary);
    function row(text) {
      const element = document.createElement('div');
      element.className = 'evidence-row';
      element.textContent = text;
      details.append(element);
    }
    if (data.data_mode === 'simulation') row('Demonstrație: traficul și zgomotul sunt simulate.');
    if (data.data_updated_at) row(`Set pregătit: ${new Date(data.data_updated_at).toLocaleString('ro-RO')}. Ora pregătirii nu este ora observației.`);
    for (const record of evidence.recommendations || []) row(`${record.park} · ${record.source} ${record.index} · observație ${record.observed_at} · senzor la ${record.sensor_distance_meters} m`);
    if (evidence.source) row(`Sursă: ${evidence.source}`);
    if (evidence.limitations) row(evidence.limitations);
    item.append(details);
  }
  function artifact(path) {
    if (typeof path !== 'string' || !/^\/(maps|visualizations)\/[a-zA-Z0-9_.-]+\.(html|png|jpg|jpeg)$/.test(path)) return;
    const frame = byId('visualization-frame');
    const image = byId('visualization-image');
    frame.hidden = image.hidden = true;
    frame.removeAttribute('src');
    image.removeAttribute('src');
    const target = path.endsWith('.html') ? frame : image;
    target.src = path;
    target.hidden = false;
    byId('download-vis').href = path;
    byId('visualization-container').hidden = false;
  }
  async function ask(query) {
    if (busy || !ready || !query.trim()) return;
    busy = true;
    send.disabled = true;
    byId('clear-chat').disabled = true;
    byId('loading').hidden = false;
    chat.setAttribute('aria-busy', 'true');
    message(query, 'user');
    input.value = '';
    byId('visualization-container').hidden = true;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 180000);
    try {
      const response = await fetch('/api/query', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query, reset_session: resetSession}), signal: controller.signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Cererea nu a putut fi procesată.');
      resetSession = false;
      const item = message(data.response);
      showEvidence(item, data);
      artifact(data.visualization);
    } catch (error) {
      message(error.name === 'AbortError' ? 'Cererea a durat prea mult. Încearcă din nou peste câteva momente.' : error.message, 'error');
    } finally {
      clearTimeout(timer);
      busy = false;
      send.disabled = !ready;
      byId('clear-chat').disabled = false;
      byId('loading').hidden = true;
      chat.removeAttribute('aria-busy');
      input.focus();
    }
  }
  byId('chat-form').addEventListener('submit', event => {event.preventDefault(); ask(input.value.trim());});
  document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => {
    input.value = button.dataset.query;
    if (ready) ask(input.value); else input.focus();
  }));
  byId('clear-chat').addEventListener('click', () => {
    resetSession = true;
    chat.replaceChildren();
    message('Conversație nouă. Ce ai vrea să explorezi?');
    byId('visualization-container').hidden = true;
    input.focus();
  });
  byId('close-vis').addEventListener('click', () => {
    byId('visualization-container').hidden = true;
    byId('visualization-frame').removeAttribute('src');
    byId('visualization-image').removeAttribute('src');
  });
  async function pollStatus() {
    if (polling || document.hidden) return;
    polling = true;
    try {
      const response = await fetch('/api/status', {signal: AbortSignal.timeout(10000)});
      if (!response.ok) throw new Error('Starea serviciului nu este disponibilă.');
      const data = await response.json();
      ready = data.ready;
      byId('data-status').textContent = ready ? (data.is_updating ? 'Date disponibile · actualizare în curs' : data.degraded ? 'Date disponibile · surse limitate' : 'Date disponibile') : 'Se pregătesc datele';
      const sourcePanel = byId('source-status');
      sourcePanel.replaceChildren();
      const labels = {parks: 'Parcuri · OpenStreetMap', air_quality: 'Aer · Airly', weather: 'Prognoză · Open-Meteo'};
      const states = {ready: 'Disponibil', loading: 'Se colectează', cached: 'Cache de rezervă', partial: 'Date parțiale', unavailable: 'Indisponibil'};
      for (const [key, title] of Object.entries(labels)) {
        const source = (data.sources || {})[key];
        const card = document.createElement('div');
        card.className = `source-card ${source?.state || 'loading'}`;
        const heading = document.createElement('strong'); heading.textContent = title;
        const state = document.createElement('span'); state.textContent = source ? (states[source.state] || source.state) : 'În așteptare';
        const description = document.createElement('small');
        description.textContent = source?.message || 'Starea sursei apare după verificare.';
        if (source?.retry_at) description.textContent += ` Reîncercare după ${new Date(source.retry_at).toLocaleString('ro-RO')}.`;
        if (key === 'air_quality' && ready) description.textContent += ` ${data.air_quality_parks || 0} parcuri cu date de aer utilizabile în setul afișat.`;
        card.append(heading, state, description); sourcePanel.append(card);
      }
      byId('parks-count').textContent = data.rows;
      byId('update-data-btn').disabled = data.is_updating;
      send.disabled = busy || !ready;
      byId('input-hint').textContent = ready ? 'Verifică sursele și ora observațiilor fiecărei recomandări.' : 'Datele se pregătesc; poți scrie deja întrebarea.';
      if (data.last_update) byId('last-update').textContent = `Pregătit: ${new Date(data.last_update).toLocaleString('ro-RO')}`;
      if (ready && data.last_update !== snapshot) {
        if (typeof data.map_url === 'string' && /^\/maps\/[a-zA-Z0-9_.-]+\.html$/.test(data.map_url)) {
          byId('parks-map').src = data.map_url;
          byId('parks-map').hidden = false;
          byId('map-placeholder')?.remove();
        }
        const sectors = data.sectors || [];
        byId('largest-sector').textContent = sectors[0]?.name || 'În așteptare';
        byId('sector-list').replaceChildren();
        for (const sector of sectors) {
          const span = document.createElement('span'); span.className = 'sector';
          span.textContent = `${sector.name} · ${sector.count}`; byId('sector-list').append(span);
        }
        snapshot = data.last_update;
      }
      notice(data.message || (!data.llm_configured ? 'Harta este disponibilă. Conversația necesită configurarea cheii Gemini pe server.' : ''));
    } catch (error) {notice('Conexiunea cu serviciul este întreruptă. Reîncercăm automat.');}
    finally {polling = false;}
  }
  byId('update-data-btn').addEventListener('click', async () => {
    byId('update-data-btn').disabled = true;
    try {
      const response = await fetch('/update-data', {method: 'POST', signal: AbortSignal.timeout(10000)});
      const data = await response.json();
      if (!response.ok) throw new Error(data.message || 'Actualizarea nu a putut porni.');
      notice(data.message);
    } catch (error) {notice(error.message); byId('update-data-btn').disabled = false;}
  });
  pollStatus();
  setInterval(pollStatus, 5000);
});
