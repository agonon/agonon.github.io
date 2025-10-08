document.addEventListener('DOMContentLoaded', () => {
  // Smooth scroll for same-page anchors (quiet & accessible)
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    const id = a.getAttribute('href').slice(1);
    const el = document.getElementById(id);
    if (el) {
      e.preventDefault();
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      history.pushState(null, '', `#${id}`);
    }
  });

  // External links open in new tab if not same origin
  document.querySelectorAll('a[href^="http"]').forEach(a => {
    try {
      const url = new URL(a.href);
      if (url.origin !== location.origin) {
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener');
      }
    } catch (_) {}
  });

  // --- Reading time + expand/collapse for interests ---
  const WPM = 150; // slower for math-y text

  function wordsIn(el) {
    const clone = el.cloneNode(true);
    clone.querySelectorAll('pre, code').forEach(n => n.remove());
    const txt = (clone.textContent || "").replace(/\s+/g, ' ').trim();
    return txt ? txt.split(' ').length : 0;
  }

  document.querySelectorAll('.interest').forEach(block => {
    const content = block.querySelector('.interest-content');
    const badge   = block.querySelector('.readtime');
    if (!content || !badge) return;
    const words = wordsIn(content);
    const mins  = Math.max(1, Math.round(words / WPM));
    badge.textContent = `${mins} min read`;
  });

  const expandAll   = document.getElementById('expand-all');
  const collapseAll = document.getElementById('collapse-all');

  if (expandAll) {
    expandAll.addEventListener('click', (e) => {
      e.preventDefault();
      document.querySelectorAll('.interest').forEach(d => d.open = true);
    });
  }
  if (collapseAll) {
    collapseAll.addEventListener('click', (e) => {
      e.preventDefault();
      document.querySelectorAll('.interest').forEach(d => d.open = false);
    });
  }
});


const btn = document.querySelector('.hamburger');
const nav = document.querySelector('#nav');
if (btn && nav) {
  btn.addEventListener('click', () => {
    const shown = nav.classList.toggle('show');
    btn.setAttribute('aria-expanded', shown ? 'true' : 'false');
  });
}

