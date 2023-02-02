if undefined(ch) then begin
  text     = ' '
  nlevels = 1981
  nx      = 932
  d = dblarr(2,/nozero)
  ch = {e:dblarr(nlevels,/nozero), g:dblarr(nlevels,/nozero), x:ptrarr(nx,/allocate)}
  openr, u, 'CH_photo_dissociation.txt', /get_lun
  readf, u, text
  for i=0,nlevels-1 do begin
    readf, u, d
    ch.e[i] = d[0]/8065.54429d0 ; wavenumber cm^-1 to eV
    ch.g[i] = d[1]
  endfor  
  readf, u, text
  up = 0
  lo = 0
  qmax = 1.0d0
  n = 0
  for i=0,nx-1 do begin
    readf, u, up, lo, qmax, n
    d = dblarr(2,n,/nozero)
    readf, u, d
;    print, i, n
    *(ch.x[i]) = {up:up, lo:lo, lam:reform(d[0,*]), q:reform(d[1,*])}
  endfor
  close, u
  free_lun, u
endif

plot, (*ch.x[0]).lam, (*ch.x[0]).q, xr=[1000,6000], yrange=[0.0, 4e-17], /xstyle, /ystyle
for i=0,931 do oplot, (*ch.x[i]).lam, (*ch.x[i]).q 
end
