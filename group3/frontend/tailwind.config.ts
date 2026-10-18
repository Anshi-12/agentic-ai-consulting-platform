import type {Config} from 'tailwindcss';
const config:Config={content:['./app/**/*.{ts,tsx}','./components/**/*.{ts,tsx}'],theme:{extend:{colors:{meridian:{navy:'#09263d',blue:'#0d6efd',cyan:'#12b8b0',lime:'#b8df63',orange:'#ff9c66',ink:'#132b3a',muted:'#71818b',line:'#dce5e9',bg:'#eef3f6'}}}} ,plugins:[]};
export default config;
