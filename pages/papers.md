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
