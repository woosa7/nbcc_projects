c
c     set SB and HB pairs as a function of time and computes their population
c
c       input  : BINARY_nres.dat
c                list_times
c                list_files    (list for SB_HB_residues.dat)
c
c       output : SB_ID_XXX_res_YYY_ZZZ_vs_time.dat
c                HB_ID_XXX_res_YYY_ZZZ_vs_time.dat
c
c                population_SB_HB.dat
c

      program population_SB_HB
      implicit none

      integer max_time   ! maximum number of times
      parameter (max_time=400)

      integer max_res    ! maximum number of protein residues
      parameter (max_res=1000)

      integer n
      real*8  time(max_time)

      integer nres_BINARY
      integer nres_monomer_1, nres_monomer_2

      integer ncontact, icontact
      character*2 ctype

c     integer ipair_SB(max_res,max_res,max_time) ! 1 if SB is formed at specific time, 0 otherwise
c     integer ipair_HB(max_res,max_res,max_time) ! 1 if HB is formed at specific time, 0 otherwise

      integer, allocatable, dimension(:,:,:) :: ipair_SB ! to avoid memory problem with max_time = 12000
      integer, allocatable, dimension(:,:,:) :: ipair_HB

      integer iflag_SB(max_res,max_res) ! 1 if SB is present somewhere in time
      integer iflag_HB(max_res,max_res) ! 1 if HB is present somewhere in time

      character*200 filename
      character*200 cid, cires, cjres, outfile
      integer ID_SB_HB

      real*8  population

      integer ires, jres, k
      integer n1, n2

      integer istart_monomer_1, iend_monomer_1
      integer istart_monomer_2, iend_monomer_2

      character*200 dummy

c
c     to avoid memory problem
c

      allocate(ipair_SB(max_res,max_res,max_time))
      allocate(ipair_HB(max_res,max_res,max_time))

c
c     set nres for BINARY complex
c

      open(10,file='BINARY_nres.dat',err=90)

      read(10,'(a)') dummy ! comment line
      read(10,*) dummy, istart_monomer_1, iend_monomer_1
      read(10,*) dummy, istart_monomer_2, iend_monomer_2

      close(10,err=90)

      nres_monomer_1 = iend_monomer_1 - istart_monomer_1 + 1
      nres_monomer_2 = iend_monomer_2 - istart_monomer_2 + 1

      nres_BINARY = nres_monomer_1 + nres_monomer_2

      if (nres_BINARY.gt.max_res)  then
        write(6,*) 'error: nres_BINARY exceeds max_res'
        stop
      endif

      write(6,*)
      write(6,*) '  number of residues in complex   : ',nres_BINARY
      write(6,*) '  number of residues in monomer 1 : ',nres_monomer_1
      write(6,*) '  number of residues in monomer 2 : ',nres_monomer_2
      write(6,*)

c
c     check the number of data in list_times and list_files
c

      open(10,file='list_times',status='old',err=90)

      n1 = 0
  100 read(10,*,end=190,err=90) time(1) ! the first time(k) variable as dummy
      n1 = n1 + 1
      goto 100
  190 continue

      close(10,err=90)

      open(11,file='list_files',status='old',err=90)

      n2 = 0
  200 read(11,'(A)',end=290,err=90) filename
      n2 = n2 + 1
      goto 200
  290 continue

      close(11,err=90)

      if(n1.ne.n2) then
        write(6,*) 'error: inconsistency in number of data'
        stop
      endif

      n = n1

      if (n.gt.max_time)  then
        write(6,*) 'error: number of data exceeds max_time'
        stop
      endif

      write(6,*) 'total number of data : ',n
      write(6,*)

c
c     initialization
c

      do ires=1,max_res
      do jres=1,max_res

        iflag_SB(ires,jres) = 0
        iflag_HB(ires,jres) = 0

        do k=1,max_time

        ipair_SB(ires,jres,k) = 0
        ipair_HB(ires,jres,k) = 0

        enddo

      enddo
      enddo

c
c    extract SB and HB data
c

      open(10,file='list_times',status='old')
      open(11,file='list_files',status='old')

      do k=1,n

        read(10,*) time(k)
        read(11,'(A)') filename

        open(20,file=filename,status='old')

        read(20,*) ncontact

        do icontact=1,ncontact

          read(20,310) ires, jres, ctype
  310     format(i3,7x,i3,5x,a2)

          if(ctype.eq.'SB') then
            ipair_SB(ires,jres,k) = 1
            if(iflag_SB(ires,jres).eq.0) iflag_SB(ires,jres)=1
          endif

          if(ctype.eq.'HB') then
            ipair_HB(ires,jres,k) = 1
            if(iflag_HB(ires,jres).eq.0) iflag_HB(ires,jres)=1
          endif

        enddo

        close(20,err=90)

      enddo

      close(10,err=90)
      close(11,err=90)

c
c     save results
c

      open(40,file='population_SB_HB.dat',err=90)

      ID_SB_HB = 0

c     SB

      do ires=1,nres_monomer_1
      do jres=nres_monomer_1+1,nres_monomer_1+nres_monomer_2

        if(iflag_SB(ires,jres).eq.1) then ! output SB_ID_XXX_res_YYY_ZZZ_vs_time.dat

          ID_SB_HB = ID_SB_HB + 1
          population = 0.0d0

          write(cid,'(i4)') ID_SB_HB + 1000 ! applicable up to 999

          write(cires,'(i4)') ires + 1000   ! applicable up to 999 residues
          write(cjres,'(i4)') jres + 1000

          outfile = 'SB_ID_'//cid(2:4)//
     .              '_res_'//cires(2:4)//'_'//cjres(2:4)//
     .              '_vs_time.dat'

          open(30,file=outfile,err=90)

          do k=1,n

            if(ipair_SB(ires,jres,k).eq.1) then
              population = population + 1.0d0
              write(30,*) time(k), ID_SB_HB
c           else
c             write(30,*) time(k), 0
            endif

          enddo

          close(30,err=90)

          population = population / dble(n) * 100.0d0 ! in %
          write(40,410) ID_SB_HB, ':', ires, jres, 'SB',population
  410     format(i3,a1,2x,i3,2x,i3,2x,a2,2x,f6.2)

        endif

      enddo
      enddo

c     HB

      do ires=1,nres_monomer_1
      do jres=nres_monomer_1+1,nres_monomer_1+nres_monomer_2

        if(iflag_HB(ires,jres).eq.1) then ! output HB_ID_XXX_res_YYY_ZZZ_vs_time.dat

          ID_SB_HB = ID_SB_HB + 1
          population = 0.0d0

          write(cid,'(i4)') ID_SB_HB + 1000 ! applicable up to 999

          write(cires,'(i4)') ires + 1000   ! applicable up to 999 residues
          write(cjres,'(i4)') jres + 1000

          outfile = 'HB_ID_'//cid(2:4)//
     .              '_res_'//cires(2:4)//'_'//cjres(2:4)//
     .              '_vs_time.dat'

          open(30,file=outfile,err=90)

          population = 0.0d0

          do k=1,n

            if(ipair_HB(ires,jres,k).eq.1) then
              population = population + 1.0d0
              write(30,*) time(k), ID_SB_HB
c           else
c             write(30,*) time(k), 0
            endif

          enddo

          close(30,err=90)

          population = population / dble(n) * 100.0d0 ! in %
          write(40,410) ID_SB_HB, ':', ires, jres, 'HB', population

        endif

      enddo
      enddo

      close(40,err=90)

c
c     dealloate memory 
c

      deallocate(ipair_SB)
      deallocate(ipair_HB)

c
   99 stop
   90 write(6,*) 'MAIN:  I/O error'
      goto 99

      end

