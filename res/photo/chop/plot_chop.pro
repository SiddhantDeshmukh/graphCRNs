if undefined(m30) then begin
  mu  = [1.0D0, 0.9195339082D0, 0.7387738651D0, 0.4779249498D0,0.1652789577D0]
  wmu = [0.0222222222D0, 0.1333059908D0, 0.2248893420D0, 0.2920426836D0, 0.3275397612D0]
  rd_rhdfine_multi2, 'd3t63g45mm00n01.0253327.fine', 5, 12, m00
  rd_rhdfine_multi2, 'd3t63g45mm30n01.0311420.fine', 5, 12, m30
  lam = m00.clam
  lamfac = !con.clight/lam^2*1e8 ; conversion to per A
  facnphoton = 1.0d-8/(!con.hplanck*!con.clight) ; conversion to photon number
  ; estimates of jnu assume no incoming radiation
  jlm00 = 0.5*(total(m00.imunu,1)#wmu)*lamfac*facnphoton
  jlm30 = 0.5*(total(m30.imunu,1)#wmu)*lamfac*facnphoton
endif  

T     = 4000.0
Trad  = 6250.0
w = findgen(1001)/1000.0*6000.0 +  500.0
!x.margin = [12.0,1.0]

plot, w, chop(T,w), thick=3, $
      xtitle='Wavelength [A]', ytitle='Photo dissociation cross section per molecule [cm!E2!N]', $
      /xstyle
oplot, w, ohop(T, w), thick=3, line=3
oplot, lam, jlm00*3e-31, color=red, psym=10, thick=2
oplot, lam, jlm30*3e-31, color=blue, psym=10, thick=2
oplot, lam, 0.5*blam(lam, trad)*facnphoton*3e-31, color=green, thick=2

print, 'lifetimes'
print, 'CH [s]'
print, 'm00: ', 1.0/(4.0*!pi*tsum(lam, chop(T,lam)*jlm00))
print, 'm30: ', 1.0/(4.0*!pi*tsum(lam, chop(T,lam)*jlm30))
print, 'OH [s]'
print, 'm00: ', 1.0/(4.0*!pi*tsum(lam, ohop(T,lam)*jlm00))
print, 'm30: ', 1.0/(4.0*!pi*tsum(lam, ohop(T,lam)*jlm30))

end  
