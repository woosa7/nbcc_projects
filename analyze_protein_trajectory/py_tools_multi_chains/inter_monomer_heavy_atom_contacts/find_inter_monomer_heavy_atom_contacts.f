c
c     find inter-monomer heavy atom contacts using the pre-defined cutoff value
c
c       input:  BINARY_natom.dat (info for number of atoms)
c               residue_info.dat (info for atoms constituting each residue in protein)
c
c               protein.pdb (complex structure)
c
c       output: heavy_atom_contacts_atoms.dat
c               heavy_atom_contacts_residues.dat
c

      program find_inter_monomer_heavy_atom_contacts
      implicit none

c     for protein

      integer max_atom ! maximum number of protein atoms
      parameter (max_atom=20000)

      integer max_res ! maximum number of protein residues
      parameter (max_res=2000)

      integer natom_BINARY, natom_BINARY_dum
      integer natom_monomer_1, natom_monomer_2

      integer istart_monomer_1, iend_monomer_1
      integer istart_monomer_2, iend_monomer_2

      integer nres_BINARY
      integer nres_monomer_1, nres_monomer_2

      real*8  x(max_atom),y(max_atom),z(max_atom)
      integer iresidue(max_atom)
      character*3 resname(max_atom)
      character*4 atomname(max_atom)

      integer istart(max_res), iend(max_res)

c     for heavy-atom contacts

      integer n_heavy_atoms

      integer, allocatable, dimension(:,:)     :: ipair_atoms ! ipair_atoms(natoms_BINARY,natoms_BINARY)
                                                              ! 1 if contact is formed, 0 otherwise
      real*8, allocatable, dimension(:,:)      :: dist_atoms  ! dist_atoms(natoms_BINARY,natoms_BINARY)

      integer n_heavy_residues

      integer ipair_residues(max_res,max_res) ! 1 if contact is formed, 0 otherwise

      real*8  cutoff  ! cutoff value for defining heavy-atom contact (4.5 here)
      real*8  cutoff2 ! cutoff2 = cutoff * cutoff

c     misc variables

      real*8  dist2, xx, yy, zz, dist_dum
      integer i, j, iatom, iatom_dum
      integer ires, jres, ires_dum

      character*200 line, dummy
      character*2 ctype_dum

      integer icontact
      integer n_check

c
c     set cutoff value
c

      cutoff  = 4.5d0 
      cutoff2 = cutoff * cutoff

c
c     read/set natom for BINARY complex
c

      open(10,file='BINARY_natom.dat',err=90)

      read(10,'(a)') dummy ! comment line
      read(10,*) dummy, istart_monomer_1, iend_monomer_1
      read(10,*) dummy, istart_monomer_2, iend_monomer_2

      close(10,err=90)

      natom_monomer_1 = iend_monomer_1 - istart_monomer_1 + 1
      natom_monomer_2 = iend_monomer_2 - istart_monomer_2 + 1

      natom_BINARY = natom_monomer_1 + natom_monomer_2

      if (natom_BINARY.gt.max_atom)  then
        write(6,*) 'error: natom_BINARY exceeds max_atom'
        stop
      endif

c
c     set residue information from 'residue_info.dat'
c

      open (10,file='residue_info.dat',status='old',err=90)

      read(10,*,err=90) nres_BINARY

      if (nres_BINARY.gt.max_res)  then
        write(6,*) 'error: nres_BINARY exceeds max_res'
        stop
      endif

      do ires=1,nres_BINARY
        read(10,*,err=90) ires_dum,istart(ires),iend(ires)
        if (ires.ne.ires_dum)  then
          write(6,*) 'error: inconsistency in ires'
          stop
        endif
      enddo

      close (10,err=90)

      ! check consistency concerning natom_BINARY

      natom_BINARY_dum = iend(nres_BINARY)

      if (natom_BINARY_dum.ne.natom_BINARY)  then
        write(6,*) 'error: inconsistency in natom_BINARY'
        stop
      endif

c
c     set nres_monomer_1 and nres_monomer_2
c

c     set nres_monomer_1 via natom_monomer_1

      nres_monomer_1 = 0 ! for checking

      do ires=1,nres_BINARY
        if(iend(ires).eq.natom_monomer_1) then
          nres_monomer_1 = ires
          exit
        endif
      enddo

      if(nres_monomer_1.eq.0) then
        write(6,*) 'error: failed to set up nres_monomer_1'
        stop
      endif

c     set nres_monomer_2

      nres_monomer_2 = nres_BINARY - nres_monomer_1

c
c     print out nres and natom for checking
c

      write(6,*)
      write(6,*) '  number of residues in complex   : ',nres_BINARY
      write(6,*) '  number of residues in monomer 1 : ',nres_monomer_1
      write(6,*) '  number of residues in monomer 2 : ',nres_monomer_2
      write(6,*)
      write(6,*) '  number of atoms in complex      : ',natom_BINARY
      write(6,*) '  number of atoms in monomer 1    : ',natom_monomer_1
      write(6,*) '  number of atoms in monomer 2    : ',natom_monomer_2
      write(6,*)

c
c     initialize ipair_atoms and ipair_residues
c

      allocate(ipair_atoms(natom_BINARY,natom_BINARY))
      allocate(dist_atoms(natom_BINARY,natom_BINARY))

      do i=1,natom_BINARY
      do j=1,natom_BINARY

        ipair_atoms(i,j) = 0

      enddo
      enddo

      do ires=1,max_res
      do jres=1,max_res

        ipair_residues(ires,jres) = 0

      enddo
      enddo

c
c     read pdb to set coordinates of protein (only for protein part)
c

      open(10,file='protein.pdb',status='old',err=90)

      do iatom=1,natom_BINARY

  100   read(10,'(a)',err=90) line
        if(line(1:4).ne.'ATOM') goto 100

        read(line,110,err=90) iatom_dum,atomname(iatom),resname(iatom),
     .                        iresidue(iatom),x(iatom),y(iatom),z(iatom)
  110   format(6x,i5,1x,a4,1x,a3,2x,i4,4x,f8.3,f8.3,f8.3)

        ! consistency check

        if(iatom_dum.ne.iatom) write(6,*) 'consistency error'

      enddo ! for do iatom=1,natom_BINARY

      close(10,err=90)

c
c     find atom pairs making inter-monomer heavy-atom contacts
c

      n_heavy_atoms = 0
      n_heavy_residues = 0

      do ires=1,nres_monomer_1
      do jres=nres_monomer_1+1,nres_monomer_1+nres_monomer_2

        do i=istart(ires),iend(ires)
        if(atomname(i)(1:1).eq.'H') cycle ! only for heavy atoms
        if(atomname(i)(2:2).eq.'H') cycle
        do j=istart(jres),iend(jres)
        if(atomname(j)(1:1).eq.'H') cycle ! only for heavy atoms
        if(atomname(j)(2:2).eq.'H') cycle

          xx = (x(i)-x(j))**2
          yy = (y(i)-y(j))**2
          zz = (z(i)-z(j))**2

          dist2 = xx + yy + zz

          if (dist2.le.cutoff2) then

            if(ipair_atoms(i,j).eq.0) then ! to avoid double counting

              n_heavy_atoms = n_heavy_atoms + 1
              ipair_atoms(i,j) = 1
              dist_atoms(i,j)  = dsqrt(dist2)

            endif

            if(ipair_residues(ires,jres).eq.0) then ! to avoid double counting

              n_heavy_residues = n_heavy_residues + 1
              ipair_residues(ires,jres) = 1

            endif

          endif ! for if (dist2.le.cutoff2) then

        enddo ! for do j=istart(jres),iend(jres)
        enddo ! for do i=istart(ires),iend(ires)

      enddo
      enddo

c
c     save inter-monomer heavy-atom contacts
c

c     atom pairs

      open(20,file='heavy_atom_contacts_atoms.dat',err=90)

      write(20,*) n_heavy_atoms

      n_check = 0

      do i=1,natom_monomer_1
      do j=natom_monomer_1+1,natom_monomer_1+natom_monomer_2

        if(ipair_atoms(i,j).eq.1) then

          n_check = n_check + 1
          write(20,210) i, atomname(i), iresidue(i), resname(i), ':',
     .                  j, atomname(j), iresidue(j), resname(j), ':',
     .                  dist_atoms(i,j)

        endif

      enddo
      enddo


  210  format(i5,1x,a4,1x,i3,1x,a3,1x,a1,1x,i5,1x,a4,1x,i3,1x,a3,
     .        1x,a1,1x,f8.3)

      close(20,err=90)

      if (n_check.ne.n_heavy_atoms)  then
        write(6,*) 'error: inconsistency in n_heavy_atoms'
        stop
      endif

c     residue pairs

      open(30,file='heavy_atom_contacts_residues.dat',err=90)

      write(30,*) n_heavy_residues

      n_check = 0

      do ires=1,nres_monomer_1
      do jres=nres_monomer_1+1,nres_monomer_1+nres_monomer_2

        i = istart(ires) ! just to get the residue name via resname(i)
        j = istart(jres) ! just to get the residue name via resname(j)

        if(ipair_residues(ires,jres).eq.1) then

          n_check = n_check + 1
          write(30,310) ires,resname(i),':',jres,resname(j)

        endif

      enddo
      enddo

  310 format(i3,1x,a3,1x,a1,1x,i3,1x,a3)

      close(30,err=90)
      close(31,err=90)
      close(32,err=90)

      if (n_check.ne.n_heavy_residues)  then
        write(6,*) 'error: inconsistency in n_heavy_residues'
        stop
      endif

      deallocate(ipair_atoms)
      deallocate(dist_atoms)

c

   99 stop
   90 write(6,*) 'MAIN:  I/O error'
      goto 99

      end
