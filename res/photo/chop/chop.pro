function chop, temp, w
; temp: temperature in K
; w; wavelength in Angström  

nw = n_elements(w)
r  = fltarr(nw,/nozero)


d1 = float(temp)
d3 = float(0.0)
pass = bytarr(3) ; 0b means by reference

for i=0,nw-1 do begin
  d2 = double(w[i]/1.0e8)            ; fortran chop expects cm

  j = call_external(getenv('IDL_SO')+'/chop_idl.so', 'chop_idl', $
      d1, d2, d3, value=pass)
  r[i] = d3
endfor
  
return, r
end 
