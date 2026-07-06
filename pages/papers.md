---
layout: page
title: Papers
permalink: /papers/
---


{%- assign papers = site.data.papers -%}

{%- comment -%} detect equal-contrib once {%- endcomment -%}
{%- assign has_equal = false -%}
{%- for paper in papers -%}
  {%- if paper.authors and paper.authors contains "*" -%}{% assign has_equal = true %}{%- endif -%}
{%- endfor -%}

<nav class="page-toc" aria-label="Page sections">
  <a href="#code">Code</a>
  <a href="#papers-list">Papers</a>
  {%- if site.data.thesis -%}<a href="#thesis">Thesis</a>{%- endif -%}
</nav>

{%- assign software = site.data.software -%}
{%- if software -%}
<h2 id="code">Code</h2>
<div class="paper-list">
  {% for item in software %}
    {% include paper_row.html paper=item %}
  {% endfor %}
</div>

<hr class="hair">
{%- endif -%}

<h2 id="papers-list">Papers</h2>
<div class="paper-list">
  {% for paper in papers %}
    {% include paper_row.html paper=paper idx=forloop.rindex %}
  {% endfor %}
</div>

{%- assign thesis = site.data.thesis -%}
{%- if thesis -%}
<hr class="hair">
<h2 id="thesis">Thesis</h2>
{%- include paper_row.html paper=thesis is_thesis=true -%}
{%- endif -%}

{%- if has_equal -%}
<p class="equal-note muted">* = equal contribution</p>
{%- endif -%}
