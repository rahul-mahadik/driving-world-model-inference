'use strict';

const copyFeedback = document.getElementById('copy-feedback');
document.querySelectorAll('[data-copy]').forEach(button => {
  const originalLabel = button.textContent;
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copy);
    if (!target) return;
    button.disabled = true;
    try {
      await navigator.clipboard.writeText(target.textContent);
      button.textContent = 'Copied';
      if (copyFeedback) copyFeedback.textContent = button.dataset.copy === 'citation-text' ? 'Software citation copied.' : 'Commands copied.';
      window.setTimeout(() => { button.textContent = originalLabel; }, 2000);
    } catch (_) {
      button.textContent = originalLabel;
      if (copyFeedback) copyFeedback.textContent = 'Copy is unavailable in this browser. Select the text in the code block to copy it manually.';
      const selection = window.getSelection();
      if (selection) {
        const range = document.createRange();
        range.selectNodeContents(target);
        selection.removeAllRanges();
        selection.addRange(range);
      }
    } finally {
      button.disabled = false;
    }
  });
});
