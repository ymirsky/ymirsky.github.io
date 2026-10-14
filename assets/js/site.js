// Header behaviour: close the mobile menu after a tap, and highlight the section in view.
document.addEventListener('DOMContentLoaded', function () {
  var toggle = document.getElementById('nav-toggle');
  var links = Array.prototype.slice.call(document.querySelectorAll('.site-nav .nav-link'));

  links.forEach(function (a) {
    a.addEventListener('click', function () { if (toggle) toggle.checked = false; });
  });

  if (!('IntersectionObserver' in window)) return;
  var bySection = {};
  links.forEach(function (a) {
    var m = /#([\w-]+)$/.exec(a.getAttribute('href') || '');
    var section = m && document.getElementById(m[1]);
    if (section) bySection[m[1]] = a;
  });
  var ids = Object.keys(bySection);
  if (!ids.length) return;

  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      links.forEach(function (a) { a.classList.remove('is-active'); });
      bySection[e.target.id].classList.add('is-active');
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  ids.forEach(function (id) { observer.observe(document.getElementById(id)); });
});
