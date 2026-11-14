'use strict';
document.querySelectorAll('[data-filter]').forEach(button => {
  button.addEventListener('click', () => {
    const filter = button.dataset.filter;
    document.querySelectorAll('[data-filter]').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    let count = 0;
    document.querySelectorAll('tr[data-kind]').forEach(row => {
      row.hidden = filter !== 'all' && row.dataset.kind !== filter;
      if (!row.hidden) count++;
    });
    document.getElementById('result-count').textContent = count + ' evidence record' + (count === 1 ? '' : 's');
  });
});
document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const feedback = document.getElementById('copy-feedback');
    try {
      await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent);
      feedback.textContent = 'Commands copied.';
    } catch (_) {
      feedback.textContent = 'Copy unavailable here. Select the commands above and copy them manually.';
    }
  });
});
