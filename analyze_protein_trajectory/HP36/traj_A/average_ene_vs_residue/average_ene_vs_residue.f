c
c     calculate the average of ene_vs_residue
c
c       input  : list_files_ene_res
c       output : average_ene_vs_residue.dat
c

      program statistics_ene_vs_residue

      implicit none

      integer maxres
      parameter (maxres=1000)

      integer nres, ndata
      real*8  ave_ene_tot(maxres)
      real*8  ave_ene_LJ(maxres)
      real*8  ave_ene_cou(maxres)

      real*8  ave2_ene_tot(maxres), sig_ene_tot

      real*8  ene_res, ene_res_LJ, ene_res_cou
      integer ires, ires_dum

      real*8  ave_ene_all

      character*200 filename

c
c     initialization
c

      ndata = 0

      do ires=1,maxres

        ave_ene_tot(ires) = 0.d0
        ave_ene_LJ(ires)  = 0.d0
        ave_ene_cou(ires) = 0.d0

        ave2_ene_tot(ires) = 0.d0

      enddo

      ave_ene_all = 0.d0

c
c     read data and caluclate the statistics
c

      open(10,file='list_files_ene_res',status='old')

  100 read(10,'(A)',end=190,err=90) filename
      open(20,file=filename,status='old',err=90)

        ndata = ndata + 1

        read(20,*,err=90) nres

        if (nres.gt.maxres)  then
          print *,'error: number of residues exceeds maxres'
          goto 99
        endif

        do ires=1,nres

          read(20,*,err=90) ires_dum,ene_res,ene_res_LJ,ene_res_cou

          if (ires_dum.ne.ires)  then
            print *,'error:  inconsistency in the residue number' 
            goto 99
          endif

          ave_ene_tot(ires) = ave_ene_tot(ires) + ene_res
          ave_ene_LJ(ires)  = ave_ene_LJ(ires)  + ene_res_LJ
          ave_ene_cou(ires) = ave_ene_cou(ires) + ene_res_cou

          ave2_ene_tot(ires) = ave2_ene_tot(ires) + ene_res*ene_res

          ave_ene_all = ave_ene_all + ene_res

        enddo

        close(20,err=90)

      goto 100

  190 continue

      close(10,err=90)

c     normalization

      do ires=1,nres

        ave_ene_tot(ires) = ave_ene_tot(ires) / ndata
        ave_ene_LJ(ires)  = ave_ene_LJ(ires)  / ndata
        ave_ene_cou(ires) = ave_ene_cou(ires) / ndata

        ave2_ene_tot(ires) = ave2_ene_tot(ires) / ndata

      enddo

      ave_ene_all = ave_ene_all / ndata

c
c     save average_ene_vs_residue.dat
c

      open(30,file='average_ene_vs_residue.dat',err=90)

      do ires=1,nres

        sig_ene_tot = dsqrt(ave2_ene_tot(ires)-ave_ene_tot(ires)**2)

        write(30,990) ires, ave_ene_tot(ires),
     .                      ave_ene_LJ(ires),
     .                      ave_ene_cou(ires),
     .                      sig_ene_tot

      enddo

      close(30,err=90)

c
c     print out summary
c

c     on screen

      write(6,*)
      write(6,*) 'total number of data used = ', ndata
      write(6,*)
      write(6,*) 'ave_ene_all         : ', ave_ene_all
      write(6,*)

c     on file

      open(30,file='summary_average_ene_vs_residue.dat',err=90)

      write(30,*)
      write(30,*) 'total number of data used = ', ndata
      write(30,*)
      write(30,*) 'ave_ene_all         : ', ave_ene_all
      write(30,*)

      close(30,err=90)

c
c     format
c

  990 format(i5,1x,f16.8,1x,f16.8,1x,f16.8,1x,f16.8)

c..........
   99 stop
   90 print *,'MAIN:  I/O error'
      goto 99

      end

