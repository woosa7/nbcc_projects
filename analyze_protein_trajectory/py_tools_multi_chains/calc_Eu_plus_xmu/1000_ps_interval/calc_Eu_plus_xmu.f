c
c       input  : energy.dat
c                xmu_vs_time.dat
c
c       output : Eu_vs_time.dat
c                Eu_plus_xmu_vs_time.dat
c

      program calc_Eu_plus_xmu
      implicit none

      real*8  time0, Eu
      real*8  time1, xmu

      real*8  dum1, dum2, dum3, dum4, dum5, dum6, dum7
      integer i, n
      integer n1, n2

      integer mod_diff

      character*120 dummy

c
c     check the number of data in energy.dat and xmu_vs_time.dat
c

      n1 = 0
      n2 = 0

      open(10,file='energy.dat',status='old',err=90)
      read(10,'(a)',err=90)     dummy ! for comment line in energy.dat
  100 read(10,*,end=190,err=90) time0, Eu, 
     .                          dum1,dum2,dum3,dum4,dum5,dum6,dum7
      n1 = n1+1
      goto 100

  190 continue
      close(10,err=90)

      open(11,file='xmu_vs_time.dat',status='old',err=90)
  200 read(11,*,end=290,err=90) time1, xmu
      n2 = n2+1
      goto 200

  290 continue
      close(11,err=90)

      if(n1.ne.n2) then
        write(6,*) 'error: inconsistency in the number of data'
        goto 99
      endif

      n = n1

      write(6,*) 'total number of data : ',n

c
c     manipulation
c

      open(10,file='energy.dat',status='old',err=90)
      open(11,file='xmu_vs_time.dat',status='old',err=90)

      open(20,file='Eu_vs_time.dat',err=90)
      open(21,file='Eu_plus_xmu_vs_time.dat',err=90)

      read(10,'(a)',err=90)  dummy ! for comment line in energy.dat

      do i=1,n

        read(10,*,err=90) time0, Eu
        read(11,*,err=90) time1, xmu

        time0 = time0*1000.d0

c       if(abs(time0-time1).gt.1.0d-3) then
c         write(6,*) 'error:  inconsistency in times',time0,time1
c         goto 99
c       endif

        mod_diff = mod(int(time0-time1),100000) ! 100000 means 100 ns

c       write(6,*) time0, time1, mod_diff

        if(mod_diff.ne.0) then
          write(6,*) 'error:  inconsistency in times',time0,time1
          goto 99
        endif

        write(20,900,err=90) time1,Eu
        write(21,900,err=90) time1,Eu+xmu

      enddo

 900  format(f16.8,1x,f16.8)

      close(10,err=90)
      close(11,err=90)

      close(20,err=90)
      close(21,err=90)

c..........
   99 stop
   90 write(6,*) 'MAIN:  I/O error'
      goto 99
c
      end

