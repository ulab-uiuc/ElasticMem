/**
 * tfr-Bench - Website JavaScript
 * Smooth scrolling, animations, and interactive elements
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {

    // Navbar burger toggle for mobile
    const $navbarBurgers = Array.prototype.slice.call(
        document.querySelectorAll('.navbar-burger'), 0
    );

    if ($navbarBurgers.length > 0) {
        $navbarBurgers.forEach(el => {
            el.addEventListener('click', () => {
                const target = el.dataset.target;
                const $target = document.getElementById(target);
                el.classList.toggle('is-active');
                $target.classList.toggle('is-active');
            });
        });
    }

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add scroll-reveal animation for sections
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all sections (except hero)
    document.querySelectorAll('.section:not(.hero)').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(section);
    });

    // Add copy functionality to BibTeX
    const bibtexPre = document.querySelector('#BibTeX pre');
    if (bibtexPre) {
        const copyButton = document.createElement('button');
        copyButton.className = 'button is-small is-dark copy-button';
        copyButton.innerHTML = '<span class="icon"><i class="fas fa-copy"></i></span>';
        copyButton.title = 'Copy to clipboard';

        const container = bibtexPre.parentElement;
        container.style.position = 'relative';
        container.insertBefore(copyButton, bibtexPre);

        copyButton.addEventListener('click', function() {
            const code = bibtexPre.querySelector('code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                copyButton.innerHTML = '<span class="icon"><i class="fas fa-check"></i></span>';
                setTimeout(() => {
                    copyButton.innerHTML = '<span class="icon"><i class="fas fa-copy"></i></span>';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy:', err);
            });
        });
    }

    // Highlight external links
    document.querySelectorAll('a[target="_blank"]').forEach(link => {
        link.setAttribute('rel', 'noopener noreferrer');
    });

    // Console message
    console.log('%ctfr-Bench 🚀', 'font-size: 20px; font-weight: bold; color: #2c3e50;');
    console.log('%cExploring tfr with Machine Learning', 'font-size: 14px; color: #666;');
    console.log('GitHub: https://github.com/kerner-lab/tfrBench');
    console.log('Paper: https://arxiv.org/abs/2510.24010');
});

function toggleReasoning() {
  var container = document.getElementById("reasoning-text-container");
  var fade = document.getElementById("reasoning-fade");
  var btn = document.getElementById("toggle-btn");
  var btnIcon = btn.querySelector("i");
  var btnText = btn.querySelector("span:last-child");

  if (container.style.maxHeight !== "none") {
    // Expand
    container.style.maxHeight = "none";
    fade.style.display = "none";
    btnText.innerText = "Show Less";
    btnIcon.className = "fas fa-angle-up";
  } else {
    // Collapse
    container.style.maxHeight = "160px"; // Adjust this value to show more/less initial text
    fade.style.display = "block";
    btnText.innerText = "Show Full Details";
    btnIcon.className = "fas fa-angle-down";
    
    // Optional: Smooth scroll back to top of box if user is far down
    // container.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}