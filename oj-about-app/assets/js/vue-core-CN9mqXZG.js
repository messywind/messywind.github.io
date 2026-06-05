import{c9 as e,y as n,T as t,ca as o,cb as c,cc as s,cd as r}from"./vendor-KDdJdyBV.js";
/**
* vue v3.5.18
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/const i=Object.create(null);e(function(e,u){if(!n(e)){if(!e.nodeType)return t;e=e.innerHTML}const a=o(e,u),m=i[a];if(m)return m;if("#"===e[0]){const n=document.querySelector(e);e=n?n.innerHTML:""}const d=c({hoistStatic:!0,onError:void 0,onWarn:t},u);d.isCustomElement||"undefined"==typeof customElements||(d.isCustomElement=e=>!!customElements.get(e));const{code:f}=s(e,d),l=new Function("Vue",f)(r);return l._rc=!0,i[a]=l});
