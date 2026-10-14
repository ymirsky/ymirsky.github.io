/*
 * Lab sync: keeps this page in step with the Offensive AI Lab site (offensive-ai-lab.com) with no
 * double bookkeeping.
 *
 *  - Featured Research: the project cards on the lab's /research/ hub are shown here as they are.
 *  - Publications: the list in index.md stays the source of truth for this site (it also has the
 *    pre-lab papers). Any paper on the lab's /publications/ page that is missing here is merged in by
 *    year, and lab project-page links are attached to matching papers.
 *
 * The lab site sends `Access-Control-Allow-Origin: *` (GitHub Pages), so the pages are read straight
 * from the browser. The last good copy is kept in localStorage so repeat visits render instantly, and
 * if the lab site can't be reached the page falls back to the static list and a plain link.
 */
(function () {
  'use strict';

  var LAB = (document.body.getAttribute('data-lab-url') || 'https://offensive-ai-lab.com').replace(/\/+$/, '');
  var CACHE_KEY = 'lab-sync:v1';
  var PUBS_COLLAPSED = 12; // papers shown before "Show all"

  var researchGrid = document.getElementById('research-grid');
  var pubSource = document.getElementById('pub-list');

  /* ---------- helpers ---------- */

  function abs(url) {
    try { return new URL(url, LAB + '/').href; } catch (e) { return url; }
  }

  function clean(s) {
    return (s || '').replace(/[‎‏]/g, '').replace(/\s+/g, ' ').trim();
  }

  function text(node) { return node ? clean(node.textContent) : ''; }

  // h('a', {class: 'x', href: '/'}, ['child', otherNode])
  function h(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (attrs[k] == null || attrs[k] === false) return;
      if (k === 'style') Object.assign(node.style, attrs[k]);
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) {
      if (c == null || c === false) return;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  }

  function readCache() {
    try { return JSON.parse(localStorage.getItem(CACHE_KEY)) || {}; } catch (e) { return {}; }
  }

  function writeCache(data) {
    try { localStorage.setItem(CACHE_KEY, JSON.stringify(data)); } catch (e) { /* private mode etc. */ }
  }

  function fetchPage(path) {
    var ctrl = 'AbortController' in window ? new AbortController() : null;
    var timer = ctrl && setTimeout(function () { ctrl.abort(); }, 10000);
    return fetch(LAB + path, { credentials: 'omit', signal: ctrl && ctrl.signal })
      .then(function (r) {
        if (!r.ok) throw new Error(path + ': HTTP ' + r.status);
        return r.text();
      })
      .then(function (html) { return new DOMParser().parseFromString(html, 'text/html'); })
      .finally(function () { clearTimeout(timer); });
  }

  /* ---------- Featured Research ---------- */

  function parseResearch(doc) {
    return Array.prototype.map.call(doc.querySelectorAll('a.rcard'), function (a) {
      var media = a.querySelector('.rcard-media');
      var style = (media && media.getAttribute('style')) || '';
      var img = /url\((['"]?)(.*?)\1\)/.exec(style);
      var pos = /background-position:\s*([^;]+)/.exec(style);
      return {
        url: abs(a.getAttribute('href')),
        image: img ? abs(img[2]) : '',
        position: pos ? pos[1].trim() : 'center',
        press: text(a.querySelector('.rcard-press')),
        venues: Array.prototype.map.call(a.querySelectorAll('.rpill'), function (p) {
          var note = text(p.querySelector('em'));
          var label = text(p);
          if (note && label.slice(-note.length) === note) label = label.slice(0, -note.length).trim();
          return { label: label, note: note, award: p.classList.contains('rpill--award') };
        }),
        title: text(a.querySelector('.rcard-title')),
        subtitle: text(a.querySelector('.rcard-subtitle')),
        summary: text(a.querySelector('.rcard-summary')),
        topics: Array.prototype.map.call(a.querySelectorAll('.rchip'), text)
      };
    }).filter(function (r) { return r.title && r.url; });
  }

  function researchCard(r, spotlight) {
    return h('a', { class: 'rcard' + (spotlight ? ' rcard--spotlight' : ''), href: r.url }, [
      h('div', { class: 'rcard-media', style: { backgroundImage: r.image ? 'url(' + JSON.stringify(r.image) + ')' : '', backgroundPosition: r.position } }, [
        r.press ? h('span', { class: 'rcard-press' }, [r.press]) : null
      ]),
      h('div', { class: 'rcard-body' }, [
        r.venues.length ? h('div', { class: 'pills' }, r.venues.map(function (v) {
          return h('span', { class: 'pill' + (v.award ? ' pill--award' : '') }, [v.label, v.note ? h('em', {}, [' ' + v.note]) : null]);
        })) : null,
        h('h3', { class: 'rcard-title' }, [r.title]),
        r.subtitle ? h('p', { class: 'rcard-subtitle' }, [r.subtitle]) : null,
        spotlight && r.summary ? h('p', { class: 'rcard-summary' }, [r.summary]) : null,
        r.topics.length ? h('div', { class: 'chips' }, r.topics.map(function (t) { return h('span', { class: 'chip' }, [t]); })) : null,
        h('span', { class: 'rcard-cta' }, ['Explore the project ', h('span', { 'aria-hidden': 'true' }, ['→'])])
      ])
    ]);
  }

  function renderResearch(list) {
    if (!researchGrid || !list || !list.length) return;
    var max = parseInt(researchGrid.getAttribute('data-max'), 10) || 4;
    var cards = list.slice(0, max).map(function (r, i) { return researchCard(r, i === 0); });
    researchGrid.replaceChildren.apply(researchGrid, cards);
    researchGrid.classList.remove('is-loading');
    var more = document.getElementById('research-more');
    if (more && list.length > max) more.firstChild.textContent = 'Browse all ' + list.length + ' projects ';
  }

  function showResearchSkeleton() {
    if (!researchGrid) return;
    var skel = [0, 1, 2, 3].map(function (i) {
      return h('div', { class: 'rcard rcard--skeleton' + (i === 0 ? ' rcard--spotlight' : ''), 'aria-hidden': 'true' }, [
        h('div', { class: 'rcard-media' }), h('div', { class: 'rcard-body' }, [h('span'), h('span'), h('span')])
      ]);
    });
    researchGrid.dataset.fallback = researchGrid.innerHTML;
    researchGrid.replaceChildren.apply(researchGrid, skel);
    researchGrid.classList.add('is-loading');
  }

  function restoreResearchFallback() {
    if (researchGrid && researchGrid.dataset.fallback) {
      researchGrid.innerHTML = researchGrid.dataset.fallback;
      researchGrid.classList.remove('is-loading');
    }
  }

  /* ---------- Publications ---------- */

  function isProjectLink(a) {
    return /project page/i.test(a.textContent) || /\/research\/[^/]+\/?$/.test(a.getAttribute('href') || '');
  }

  // One <li> from a Markdown publication list -> {title, url, details, badges, project, year}
  function parsePub(li, base) {
    var links = li.querySelectorAll('a');
    if (!links.length) return null;
    var titleLink = links[0];
    var project = null;
    var rest = li.cloneNode(true);
    var restLinks = rest.querySelectorAll('a');
    restLinks[0].remove();
    for (var i = 1; i < links.length; i++) {
      if (isProjectLink(links[i])) {
        project = project || base(links[i].getAttribute('href'));
        restLinks[i].remove();
      }
    }
    var badges = Array.prototype.map.call(rest.querySelectorAll('code'), text);
    Array.prototype.forEach.call(rest.querySelectorAll('code'), function (c) { c.remove(); });
    var details = clean(rest.textContent).replace(/^[\s.,;:]+/, '').replace(/[\s&,;]+$/, '');
    var years = (details.match(/\b(19[89]\d|20\d\d)\b/g) || []).map(Number);
    return {
      title: text(titleLink).replace(/[.\s]+$/, ''),
      url: base(titleLink.getAttribute('href')),
      details: details,
      badges: badges,
      project: project,
      year: years.length ? Math.max.apply(null, years) : null
    };
  }

  function parsePubList(ul, base) {
    if (!ul) return [];
    return Array.prototype.map.call(ul.querySelectorAll(':scope > li'), function (li) { return parsePub(li, base); })
      .filter(Boolean);
  }

  function parseLabPubs(doc) {
    var heading = doc.getElementById('academic-articles');
    var el = heading && heading.nextElementSibling;
    while (el && el.tagName !== 'UL' && !/^H[1-3]$/.test(el.tagName)) el = el.nextElementSibling;
    return el && el.tagName === 'UL' ? parsePubList(el, abs) : [];
  }

  function key(title) {
    return title.toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '');
  }

  // Lab papers missing from the local list go at the top of their year (after any lab papers already
  // placed there, so the lab's newest-first order is kept).
  function mergePubs(local, lab) {
    var byKey = {};
    local.forEach(function (p) { byKey[key(p.title)] = p; });
    var out = local.slice();
    (lab || []).forEach(function (src) {
      var p = Object.assign({}, src);
      var hit = byKey[key(p.title)];
      if (hit) {
        if (!hit.project && p.project) hit.project = p.project;
        return;
      }
      p.fromLab = true;
      var at = p.year ? out.findIndex(function (q) {
        return q.year && q.year <= p.year && !(q.fromLab && q.year === p.year);
      }) : 0;
      out.splice(at < 0 ? out.length : at, 0, p);
      byKey[key(p.title)] = p;
    });
    return out;
  }

  var ME = /\b(Yisroel Mirsky|Y\.? Mirsky)\b/;

  function detailsNode(details) {
    var parts = details.split(ME);
    return h('p', { class: 'pub-meta' }, parts.map(function (part, i) {
      return i % 2 ? h('strong', {}, [part]) : part;
    }));
  }

  function badgeClass(b) {
    if (/award/i.test(b)) return 'badge badge--award';
    if (/\bA\*|\bRank A\b/.test(b)) return 'badge badge--rank';
    return 'badge';
  }

  var FILTERS = [
    { id: 'all', label: 'All', test: function () { return true; } },
    { id: 'top', label: 'A* venues', test: function (p) { return p.badges.some(function (b) { return /A\*/.test(b); }); } },
    { id: 'award', label: 'Awards', test: function (p) { return p.badges.some(function (b) { return /award/i.test(b); }); } },
    { id: 'project', label: 'Project pages', test: function (p) { return !!p.project; } }
  ];

  var pubState = { filter: 'all', expanded: false, items: [] };
  var pubRoot = null;

  function renderPubs(items) {
    if (!pubSource) return;
    pubState.items = items;
    if (!pubRoot) {
      pubRoot = h('div', { class: 'pubs' });
      pubSource.parentNode.insertBefore(pubRoot, pubSource);
      pubSource.hidden = true;
    }

    var filter = FILTERS.filter(function (f) { return f.id === pubState.filter; })[0] || FILTERS[0];
    var matches = items.filter(filter.test);
    var shown = pubState.expanded ? matches : matches.slice(0, PUBS_COLLAPSED);

    var toolbar = h('div', { class: 'pub-toolbar', role: 'toolbar', 'aria-label': 'Filter publications' }, FILTERS.map(function (f) {
      var n = items.filter(f.test).length;
      if (!n) return null;
      var btn = h('button', { type: 'button', class: 'filter' + (f.id === filter.id ? ' is-active' : ''), 'aria-pressed': String(f.id === filter.id) }, [
        f.label, h('span', { class: 'filter-count' }, [String(n)])
      ]);
      btn.addEventListener('click', function () {
        pubState.filter = f.id;
        pubState.expanded = false;
        renderPubs(pubState.items);
      });
      return btn;
    }));

    var groups = [];
    shown.forEach(function (p) {
      var last = groups[groups.length - 1];
      var label = p.year ? String(p.year) : '';
      if (!last || last.label !== label) groups.push(last = { label: label, items: [] });
      last.items.push(p);
    });

    var list = h('div', { class: 'pub-groups' }, groups.map(function (g) {
      return h('div', { class: 'pub-group' }, [
        h('h3', { class: 'pub-year' }, [g.label]),
        h('ol', { class: 'pub-items' }, g.items.map(function (p) {
          return h('li', { class: 'pub' }, [
            h('a', { class: 'pub-title', href: p.url }, [p.title]),
            p.details ? detailsNode(p.details) : null,
            (p.badges.length || p.project) ? h('div', { class: 'pub-badges' }, p.badges.map(function (b) {
              return h('span', { class: badgeClass(b) }, [b]);
            }).concat(p.project ? [h('a', { class: 'badge badge--link', href: p.project }, ['Project page ', h('span', { 'aria-hidden': 'true' }, ['→'])])] : [])) : null
          ]);
        }))
      ]);
    }));

    var more = null;
    if (matches.length > PUBS_COLLAPSED) {
      more = h('button', { type: 'button', class: 'btn pub-more' }, [
        pubState.expanded ? 'Show fewer' : 'Show all ' + matches.length + ' publications'
      ]);
      more.addEventListener('click', function () {
        var wasExpanded = pubState.expanded;
        pubState.expanded = !pubState.expanded;
        renderPubs(pubState.items);
        if (wasExpanded) pubRoot.scrollIntoView({ block: 'start' });
      });
    }

    pubRoot.replaceChildren(toolbar, list);
    if (more) pubRoot.appendChild(h('div', { class: 'pub-more-wrap' }, [more]));
  }

  /* ---------- go ---------- */

  var localPubs = pubSource ? parsePubList(pubSource.querySelector('ul'), function (u) { return u; }) : [];
  var cache = readCache();

  if (cache.research && cache.research.length) renderResearch(cache.research);
  else showResearchSkeleton();
  renderPubs(mergePubs(localPubs, cache.pubs));

  Promise.allSettled([fetchPage('/research/'), fetchPage('/publications/')]).then(function (res) {
    var research = res[0].status === 'fulfilled' ? parseResearch(res[0].value) : null;
    var pubs = res[1].status === 'fulfilled' ? parseLabPubs(res[1].value) : null;
    var next = {
      research: research && research.length ? research : cache.research,
      pubs: pubs && pubs.length ? pubs : cache.pubs,
      at: Date.now()
    };

    if (next.research && next.research.length) {
      if (JSON.stringify(next.research) !== JSON.stringify(cache.research)) renderResearch(next.research);
    } else {
      restoreResearchFallback();
    }
    if (JSON.stringify(next.pubs) !== JSON.stringify(cache.pubs)) {
      localPubs = parsePubList(pubSource && pubSource.querySelector('ul'), function (u) { return u; });
      renderPubs(mergePubs(localPubs, next.pubs));
    }
    if (research || pubs) writeCache(next);
    res.forEach(function (r) { if (r.status === 'rejected' && window.console) console.warn('lab-sync:', r.reason); });
  });
})();
