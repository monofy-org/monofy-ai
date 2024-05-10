export async function triggerActive(elt: HTMLElement, timeout = 100) {  
  elt.classList.toggle("active", true);
  setTimeout(() => {
    elt.classList.toggle("active", false);
  }, timeout);
}
