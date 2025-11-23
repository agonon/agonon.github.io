---
layout: home
title: "Bonjour — I'm Antoine"
seo_title: "Antoine Gonon, mathematics of deep learning"
---

<div class="hero-grid intro">
  <img class="avatar" src="/assets/img/me.jpg" alt="Antoine Gonon" width="144" height="144" loading="lazy" decoding="async">

  <div>
    <p>I'm broadly interested in the mathematical and algorithmic aspects of deep learning.</p>
    <ul class="roles">
      <li>
        <div class="role">
            <span>
              Postdoc @
              <a href="{{ site.links.epfl }}" target="_blank" rel="noopener">EPFL</a>
              (with
              <a href="{{ site.people.boumal.url }}" target="_blank" rel="noopener">{{ site.people.boumal.name }}</a>)
            </span>
            <span class="dates">Dec 2024 – present</span>
          </div>
        </li>
      <li>
        <div class="role">
          <span>
            PhD @
            <a href="{{ site.links.enslyon }}" target="_blank" rel="noopener">ENS Lyon</a>
            (with
            <a href="{{ site.people.gribonval.url }}" target="_blank" rel="noopener">{{ site.people.gribonval.name }}</a>,
            <a href="{{ site.people.riccietti.url }}" target="_blank" rel="noopener">{{ site.people.riccietti.name }}</a>,
            <a href="{{ site.people.brisebarre.url }}" target="_blank" rel="noopener">{{ site.people.brisebarre.name }}</a>)
          </span>
          <span class="dates">2021 – 2024</span>
        </div>
      </li>
    </ul>
    <div class="cta-row">
      <a class="btn" href="/interests/" id="btn-interests">What I work on</a>
      <a class="btn ghost" href="/papers/" id="btn-papers" style="margin-left:8px;">Latest research</a>
    </div>
  </div>
</div>


<div id="dynamic-sections" class="dynamic-sections" aria-live="polite"></div>

<div id="tpl-interests" hidden>
  {% include btn-interests.html compact=true %}
</div>

<div id="tpl-papers" hidden>
  {% include btn-papers.html %}
</div>

<script>
(function() {
  var root = document.getElementById('dynamic-sections');
  var btnInterests = document.getElementById('btn-interests');
  var btnPapers = document.getElementById('btn-papers');

  function showTemplate(tplId) {
    var tpl = document.getElementById(tplId);
    if (!tpl) return;
    root.innerHTML = tpl.innerHTML; // replace content each time
    // optional: scroll into view if the hero is tall
    root.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  btnInterests.addEventListener('click', function(e) {
    // with JS: prevent navigation and swap content
    e.preventDefault();
    showTemplate('tpl-interests');
  });

  btnPapers.addEventListener('click', function(e) {
    e.preventDefault();
    showTemplate('tpl-papers');
  });
})();
</script>